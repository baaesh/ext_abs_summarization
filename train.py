import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from modules.seq2seq import Seq2Seq


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


def train(opt, data):
    print('Loading GloVe pretrained vectors')
    glove_embeddings = load_pth(opt['glove_path'])

    device = torch.device(opt['device'])
    model = Seq2Seq(opt=opt,
                    pad_id=data.vocab.pad_id,
                    bos_id=data.vocab.bos_id,
                    vectors=glove_embeddings).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    print('Training Start!')
    for batch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(batch+1))
        for step, batch in enumerate(data.train_loader):
            try:
                model.train()

                batch = to_device(batch, device=device)
                logits = model(batch['extracted']['words'],
                               batch['extracted']['length'],
                               batch['abstract']['words'],
                               batch['abstract']['length'])
                targets = batch['abstract']['words']

                batch_loss = 0
                for i in range(logits.size(1) - 1):
                    batch_loss += criterion(logits[:, i], targets[:, i+1])
                loss += batch_loss.item() / (targets.size(0) * (targets.size(1) - 1))

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if (step + 1) % 100 == 0:
                    print('step ' + str(step+1) +': loss '+ str(loss))
                    loss = 0
            except Exception as e:
                print(batch['extracted']['length'].max())
                print(batch['abstract']['length'].max())
                print(e)
                return


if __name__ == '__main__':
    opt = set_args()
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)