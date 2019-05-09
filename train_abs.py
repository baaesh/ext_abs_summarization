import copy
import os
from time import gmtime, strftime

from nltk import sent_tokenize
import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from modules.abstractor import Seq2Seq, PointerGenerator
from modules.utils import idx2origin, strip_sequence
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


def save_checkpoint(opt, save_dict, max_rougeL):
    model_name_str = 'Abstractor_'

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_name = f'{model_name_str}_{opt["model_time"]}_{max_rougeL:.4f}.pth'
    torch.save(save_dict, 'saved_models/' + model_name)


def validate(step, model, data_loader, device):
    rouge1_sum, rouge2_sum, rougeL_sum = 0, 0, 0
    count = 0
    for _, batch in enumerate(data_loader):
        model.eval()
        batch = to_device(batch, device=device)
        batch_size = len(batch['id'])

        preds = model(batch['extracted']['text_unk'],
                      batch['extracted']['text'],
                      batch['extracted']['len']).cpu().numpy()
        golds = batch['abstract']['origin']
        for i in range(batch_size):
            pred = strip_sequence(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
            pred_text = idx2origin(pred, data.vocab, batch['oov_tokens'][i])
            eval = sent_tokenize(pred_text)
            ref = golds[i]

            rouge1_sum += rouge.rouge_n(eval, ref, n=1)['f']
            rouge2_sum += rouge.rouge_n(eval, ref, n=2)['f']
            rougeL_sum += rouge.rouge_l_summary_level(eval, ref)['f']
            count += 1

    print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) +
          ': ROUGE-1 ' + str(rouge1_sum / count) +
          ' ROUGE-2 ' + str(rouge2_sum / count) +
          ' ROUGE-L ' + str(rougeL_sum / count))
    return rougeL_sum / count


def train(opt, data):
    device = torch.device(opt['device'])
    model = PointerGenerator(opt=opt,
                             pad_id=data.vocab.pad_id,
                             bos_id=data.vocab.bos_id,
                             unk_id=data.vocab.unk_id,
                             vectors=data.vectors).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    criterion = nn.NLLLoss()

    max_rougeL = 0
    best_model, best_opt, best_epoch = None, None, 0
    print('Training Start!')
    for epoch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(epoch + 1))
        for step, batch in enumerate(data.train_loader):
            model.train()
            batch = to_device(batch, device=device)
            logits = model(batch['extracted']['text_unk'],
                           batch['extracted']['text'],
                           batch['extracted']['len'],
                           batch['abstract']['text_unk'],
                           batch['abstract']['text'],
                           batch['abstract']['len'])
            targets = batch['abstract']['text']

            batch_loss = 0
            for i in range(len(logits) - 1):
                batch_loss += criterion(logits[i], targets[:, i + 1])
            loss += batch_loss.item() / (targets.size(0) * (targets.size(1) - 1))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) + ': loss ' + str(loss))
                loss = 0
            if (step + 1) % opt['validate_every'] == 0:
                rougeL = validate(step, model, data.valid_loader, device)
                if max_rougeL < rougeL:
                    max_rougeL = rougeL
                    best_model = copy.deepcopy(model.state_dict())
                    best_opt = copy.deepcopy(optimizer.state_dict())
                    best_epoch = epoch
    save_checkpoint(opt,
                    {'epoch': best_epoch,
                     'model_state_dict': best_model,
                     'optimizer_state_dict': best_opt},
                    max_rougeL)


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'a'
    opt['model_time'] = strftime('%H:%M:%S', gmtime())
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
