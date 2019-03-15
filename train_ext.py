import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
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


if __name__ == '__main__':
    opt = set_args()
    opt['mode'] = 'e'
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)
