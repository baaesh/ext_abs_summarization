import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from metric import f1_score
from modules.extractor import PointerNetwork


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


def strip(idx_list, max_len, pad_id):
    end_idx = max_len
    for i in range(len(idx_list)):
        if idx_list[i] == pad_id:
            end_idx = i
            break
    return idx_list[:end_idx]


def train(opt, data):
    print('Loading GloVe pretrained vectors')
    glove_embeddings = load_pth(opt['glove_path'])

    device = torch.device(opt['device'])
    model = PointerNetwork(opt=opt,
                           pad_id=data.vocab.pad_id,
                           vectors=glove_embeddings).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    criterion = nn.NLLLoss()

    print('Training Start!')
    for epoch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(epoch + 1))
        for step, batch in enumerate(data.train_loader):
            model.train()
            batch = to_device(batch, device=device)

            logits = model(batch['article']['sentences'],
                           batch['article']['length'],
                           batch['target']['positions'].long(),
                           batch['target']['length'])
            targets = batch['target']['positions'].long()

            batch_loss = 0
            for i in range(targets.size(1)):
                batch_loss += criterion(logits[:, i], targets[:, i])
            loss += batch_loss.item() / (targets.size(0) * (targets.size(1)))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) + ': loss ' + str(loss))
                loss = 0
            if (step + 1) % opt['validate_every'] == 0:
                f1_sum = 0
                prec_sum = 0
                rec_sum = 0
                count = 0
                for _, batch in enumerate(data.valid_loader):
                    model.eval()
                    batch = to_device(batch, device=device)
                    preds = model(batch['article']['sentences'],
                                  batch['article']['length']).cpu().numpy()
                    golds = batch['target']['positions'].cpu().numpy()
                    for i in range(len(golds)):
                        pred = strip(preds[i], len(preds[i]), data.vocab.pad_id)
                        gold = strip(golds[i], len(golds[i]), data.vocab.pad_id)
                        f1, prec, rec = f1_score(pred, gold)
                        f1_sum += f1
                        prec_sum += prec
                        rec_sum += rec
                        count += 1

                print('step ' + str(step + 1) + '/' + str(len(data.train_loader)) +
                      ': F1 ' + str(f1_sum / count) +
                      ' Precision ' + str(prec_sum / count) +
                      ' Recall ' + str(rec_sum / count))


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'e'
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
