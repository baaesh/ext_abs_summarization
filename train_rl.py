import copy
import os
from time import gmtime, strftime

from nltk import sent_tokenize
import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from modules.rl import ActorCritic
from modules.abstractor import PointerGenerator
from modules.utils import point2text, point2result, one_hot_embedding, idx2origin, strip_sequence
import rouge


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


def a2c_loss(points, logits, rewards, scores):
    batch_size, max_ext, num_sentence = logits.size()
    points = points.view(-1)
    logits = logits.view(-1, num_sentence)

    points_one_hot = one_hot_embedding(points, batch_size * max_ext, num_sentence, logits.device)
    log_probs = torch.masked_select(logits, points_one_hot.byte()).view(batch_size, -1)
    advantages = rewards - scores
    loss = - (log_probs * advantages).mean()
    return loss


def validate(step, extractor, abstractor, data_loader, device):
    rouge1_sum = 0
    rouge2_sum = 0
    rougeL_sum = 0
    count = 0
    for _, batch in enumerate(data_loader):
        extractor.eval()
        batch = to_device(batch, device=device)
        batch_size = len(batch['id'])

        (points, logits), scores = extractor(batch['article']['sents_unk'],
                                             batch['article']['lens'])

        ext_unk, ext_len = point2text(points, batch['article']['sents_unk'],
                                      data.vocab.pad_id, device)
        ext, _ = point2text(points,
                            batch['article']['sents'],
                            data.vocab.pad_id, device)
        with torch.no_grad():
            abstractor.eval()
            preds = abstractor(ext_unk, ext, ext_len).cpu().numpy()
            golds = batch['abstract']['origin']
            exts = point2result(points.cpu().numpy(), batch['article']['origin'])

        for i in range(batch_size):
            pred = strip_sequence(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
            pred_text = idx2origin(pred, data.vocab, batch['oov_tokens'][i])
            eval = sent_tokenize(pred_text)
            ref = golds[i]
            #if i == 0:
            #    print(exts[i])
            #    print(eval)
            #    print(ref)
            rouge1_sum += rouge.rouge_n(eval, ref, n=1)['f']
            rouge2_sum += rouge.rouge_n(eval, ref, n=2)['f']
            rougeL_sum += rouge.rouge_l_summary_level(eval, ref)['f']
            count += 1
    print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) +
          ': ROUGE-1 ' + str(rouge1_sum / count) +
          ' ROUGE-2 ' + str(rouge2_sum / count) +
          ' ROUGE-L ' + str(rougeL_sum / count))


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
        critic_loss = 0
        rl_loss = 0
        print("Epoch " + str(epoch + 1))
        for step, batch in enumerate(data.train_loader):
            extractor.train()
            batch = to_device(batch, device=device)
            batch_size = len(batch['id'])

            # points: batch_size x max_ext
            # logits: batch_size x max_ext x num_sentence
            (points, logits), scores = extractor(batch['article']['sents_unk'],
                                                 batch['article']['lens'])

            points = points.detach()
            ext_unk, ext_len = point2text(points, batch['article']['sents_unk'],
                                          -1, device)
            ext, _ = point2text(points, batch['article']['sents'],
                                -1, device)

            with torch.no_grad():
                abstractor.eval()
                preds = abstractor(ext_unk, ext, ext_len).cpu().numpy()
                golds = batch['abstract']['origin']

            rewards = []
            for i in range(batch_size):
                pred = strip_sequence(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
                pred_text = idx2origin(pred, data.vocab, batch['oov_tokens'][i])
                eval = sent_tokenize(pred_text)
                ref = golds[i]
                rewards.append(rouge.rouge_l_summary_level(eval, ref)['f'])
            rewards = to_device(torch.tensor(rewards).unsqueeze(-1).expand_as(scores).contiguous(), device)

            batch_critic_loss = critic_criterion(scores, rewards)
            critic_loss += batch_critic_loss.item()
            batch_rl_loss = a2c_loss(points, logits, rewards, scores.detach())
            rl_loss += batch_rl_loss.item()
            batch_loss = batch_critic_loss + batch_rl_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step + 1) +
                      '/' + str(len(data.train_loader)) +
                      ': critic loss ' + str(critic_loss) +
                      ' rl loss ' + str(rl_loss))
                critic_loss = 0
                rl_loss = 0
            if (step + 1) % opt['validate_every'] == 0:
                validate(step, extractor, abstractor, data.valid_loader, device)


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'r'
    opt['model_time'] = strftime('%H:%M:%S', gmtime())
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
