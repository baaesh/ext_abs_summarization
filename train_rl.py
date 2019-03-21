import copy
import os
from time import gmtime, strftime

import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from metric import rouge_L, rouge_n
from modules.rl import ActorCritic
from modules.abstractor import PointerGenerator
from modules.utils import point2text, one_hot_embedding, idx2origin, remove_pad


def load_pth(path):
    pth_data = torch.load(path)
    return pth_data


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: to_device(v, device=device) for k, v in batch.items()}
    else:
        return batch


def strip(words_list, max_len, bos_id, eos_id):
    start_idx = 0
    if words_list[0] == bos_id:
        start_idx = 1
    end_idx = max_len
    for i in range(len(words_list)):
        if words_list[i] == eos_id:
            end_idx = i
            break
    return words_list[start_idx:end_idx]


def a2c_loss(points, logits, rewards, scores):
    batch_size, max_ext, num_sentence = logits.size()
    points = points.view(-1)
    logits = logits.view(-1, num_sentence)

    points_one_hot = one_hot_embedding(points, batch_size * max_ext, num_sentence, logits.device)
    log_probs = - torch.masked_select(logits, points_one_hot.byte()).view(batch_size, -1)
    advantages = rewards - scores
    loss = (log_probs * advantages).mean()
    return loss


def train(opt, data):
    print('Loading Pre-Trained Models')
    ext_checkpoint = load_pth(opt['extractor_path'])
    abs_checkpoint = load_pth(opt['abstractor_path'])

    device = torch.device(opt['device'])
    extractor = ActorCritic(opt=opt,
                            pad_id=data.vocab.pad_id,
                            ext_state_dict=ext_checkpoint['model_state_dict']).to(device)
    abstractor = PointerGenerator(opt=opt,
                                  pad_id=data.vocab.pad_id,
                                  bos_id=data.vocab.bos_id,
                                  unk_id=data.vocab.unk_id).to(device)
    abstractor.load_state_dict(abs_checkpoint['model_state_dict'])

    parameters = filter(lambda p: p.requires_grad, extractor.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    critic_criterion = nn.MSELoss()

    print('Training Start!')
    for epoch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(epoch + 1))
        for step, batch in enumerate(data.train_loader):
            extractor.train()
            batch = to_device(batch, device=device)

            # points: batch_size x max_ext
            # logits: batch_size x max_ext x num_sentence
            (points, logits), scores = extractor(batch['article']['sentences'],
                                                 batch['article']['num_sentence'],
                                                 batch['article']['length'])

            points = points.detach()
            extracted, length = point2text(points, batch['article']['sentences_origin'],
                                           batch['article']['length'],
                                           data.vocab.pad_id, device)
            extracted_extended, _ = point2text(points, batch['article']['sentences_extended_origin'],
                                               batch['article']['length'],
                                               data.vocab.pad_id, device)

            with torch.no_grad():
                abstractor.eval()
                preds = abstractor(extracted, extracted_extended, length).cpu().numpy()
                golds = batch['abstract']['words_extended'].cpu().numpy()

            rewards = []
            for i in range(len(golds)):
                pred = strip(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
                gold = strip(golds[i], len(golds[i]), data.vocab.bos_id, data.vocab.eos_id)
                rewards.append(rouge_L(pred, gold))
            rewards = to_device(torch.tensor(rewards).unsqueeze(-1).expand_as(scores).contiguous(), device)

            batch_loss = critic_criterion(scores, rewards)
            batch_loss += a2c_loss(points, logits, rewards, scores.detach())
            loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) + ': loss ' + str(loss))
                loss = 0
            if (step) % opt['validate_every'] == 0:
                rouge1_sum = 0
                rouge2_sum = 0
                rougeL_sum = 0
                count = 0
                for _, batch in enumerate(data.valid_loader):
                    extractor.eval()
                    batch = to_device(batch, device=device)

                    (points, logits), scores = extractor(batch['article']['sentences'],
                                                         batch['article']['num_sentence'],
                                                         batch['article']['length'])

                    extracted, length = point2text(points, batch['article']['sentences_origin'],
                                                   batch['article']['length'],
                                                   data.vocab.pad_id, device)
                    extracted_extended, _ = point2text(points, batch['article']['sentences_extended_origin'],
                                                       batch['article']['length'],
                                                       data.vocab.pad_id, device)

                    with torch.no_grad():
                        abstractor.eval()
                        preds = abstractor(extracted, extracted_extended, length).cpu().numpy()
                        golds = batch['abstract']['words_extended'].cpu().numpy()
                        exts = extracted_extended.cpu().numpy()

                    for i in range(len(golds)):
                        ext = strip(exts[i], len(exts[i]), data.vocab.bos_id, data.vocab.eos_id)
                        ext = remove_pad(ext, data.vocab.pad_id)
                        pred = strip(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
                        gold = strip(golds[i], len(golds[i]), data.vocab.bos_id, data.vocab.eos_id)
                        if i == 0:
                            ext_origin = idx2origin(ext, data.vocab, batch['oov_tokens'][i])
                            pred_origin = idx2origin(pred, data.vocab, batch['oov_tokens'][i])
                            gold_origin = idx2origin(gold, data.vocab, batch['oov_tokens'][i])
                            print('ext: \n' + ext_origin)
                            print('pred: \n' + pred_origin)
                            print('gold: \n' + gold_origin)
                        rouge1_sum += rouge_n(pred, gold, n=1)
                        rouge2_sum += rouge_n(pred, gold, n=2)
                        rougeL_sum += rouge_L(pred, gold)
                        count += 1

                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) +
                      ': ROUGE-1 ' + str(rouge1_sum / count) +
                      ' ROUGE-2 ' + str(rouge2_sum / count) +
                      ' ROUGE-L ' + str(rougeL_sum / count))


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'r'
    opt['model_time'] = strftime('%H:%M:%S', gmtime())
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
