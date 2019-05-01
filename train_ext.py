import copy
import os
import math
from time import gmtime, strftime

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from config import set_args
from data import CnnDm
from metric import f1_score
from modules.extractor import PointerNetwork, HierarchicalPointerNetwork
from modules.utils import point2result, strip_positions
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


def save_checkpoint(opt, save_dict, max_dev_f1):
    model_name_str = 'Extractor_'

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_name = f'{model_name_str}_{opt["model_time"]}_{max_dev_f1:.4f}.pth'
    torch.save(save_dict, 'saved_models/' + model_name)


def sequence_loss(logits, targets, criterion, pad_idx=-1):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask).view(-1)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    loss = criterion(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


def validate(step, model, data_loader, criterion, device):
    f1_sum, prec_sum, rec_sum = 0, 0, 0
    rouge1_sum, rouge2_sum, rougeL_sum = 0, 0, 0
    count = 0
    loss = 0
    batch_count = 0
    for _, batch in enumerate(data_loader):
        model.eval()
        batch = to_device(batch, device=device)
        batch_size = len(batch['id'])

        (preds, logits), _ = model(batch['article']['sents_unk'],
                                   batch['article']['lens'])

        preds = preds.cpu().numpy()
        results = point2result(preds, batch['article']['origin'])
        golds = batch['abstract']['origin']

        # validation loss
        targets = batch['target']['position'].long()[:, :4]
        loss += sequence_loss(logits, targets, criterion, pad_idx=-1).item()
        batch_count += 1

        targets = batch['target']['position'].long().cpu().numpy()
        for i in range(batch_size):
            # point level evaluation
            pred = preds[i]
            target = targets[i]
            f1, prec, rec = f1_score(pred, target)
            f1_sum += f1
            prec_sum += prec
            rec_sum += rec

            # summary level evaluation
            eval = results[i]
            ref = golds[i]
            rouge1_sum += rouge.rouge_n(eval, ref, n=1)['f']
            rouge2_sum += rouge.rouge_n(eval, ref, n=2)['f']
            rougeL_sum += rouge.rouge_l_summary_level(eval, ref)['f']
            count += 1
    f1_avg = f1_sum / count
    prec_avg = prec_sum / count
    rec_avg = rec_sum / count
    print('validation loss: ' + str(loss / batch_count))
    print('step %d/%d: F1 %.4f Precision %.4f Recall %.4f' %
          (step + 1, len(data.train_loader),
           f1_avg, prec_avg, rec_avg))
    print(' ROUGE-1 ' + str(rouge1_sum / count) +
          ' ROUGE-2 ' + str(rouge2_sum / count) +
          ' ROUGE-L ' + str(rougeL_sum / count))
    return f1_avg


def train(opt, data):
    device = torch.device(opt['device'])
    model = PointerNetwork(opt=opt,
                           pad_id=data.vocab.pad_id,
                           vectors=data.vectors).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    scheduler = StepLR(optimizer, step_size=1, gamma=opt['lr_gamma'])
    criterion = nn.CrossEntropyLoss()

    max_dev_f1 = 0
    best_model, best_opt, best_epoch = None, None, 0
    print('Training Start!')
    for epoch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(epoch + 1))
        scheduler.step()
        for step, batch in enumerate(data.train_loader):
            model.train()
            batch = to_device(batch, device=device)

            logits = model(batch['article']['sents_unk'],
                           batch['article']['lens'],
                           batch['target']['position'],
                           batch['target']['len'])
            targets = batch['target']['position']

            batch_loss = sequence_loss(logits, targets, criterion, pad_idx=0)
            loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=opt['norm_limit'])
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) + ': loss ' + str(
                    loss / opt['print_every']))
                loss = 0
            if (step + 1) % opt['validate_every'] == 0:
                f1_avg = validate(step, model, data.valid_loader, criterion, device)
                if max_dev_f1 < f1_avg:
                    max_dev_f1 = f1_avg
                    best_model = copy.deepcopy(model.state_dict())
                    best_opt = copy.deepcopy(optimizer.state_dict())
                    best_epoch = epoch

    save_checkpoint(opt,
                    {'epoch': best_epoch,
                     'model_state_dict': best_model,
                     'optimizer_state_dict': best_opt},
                    max_dev_f1)


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'e'
    opt['model_time'] = strftime('%H:%M:%S', gmtime())
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
