import torch
from torch import nn, optim

from config import set_args
from data import CnnDm
from metric import rouge_L, rouge_n
from modules.abstractor import Seq2Seq, PointerGenerator


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
    return words_list[start_idx:end_idx]

def train(opt, data):
    print('Loading GloVe pretrained vectors')
    glove_embeddings = load_pth(opt['glove_path'])

    device = torch.device(opt['device'])
    model = PointerGenerator(opt=opt,
                    pad_id=data.vocab.pad_id,
                    bos_id=data.vocab.bos_id,
                    unk_id=data.vocab.unk_id,
                    vectors=glove_embeddings).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
    criterion = nn.NLLLoss()

    print('Training Start!')
    for epoch in range(opt['epochs']):
        loss = 0
        print("Epoch " + str(epoch+1))
        for step, batch in enumerate(data.train_loader):
            model.train()
            batch = to_device(batch, device=device)
            logits = model(batch['extracted']['words'],
                           batch['extracted']['words_extended'],
                           batch['extracted']['length'],
                           batch['abstract']['words'],
                           batch['abstract']['words_extended'],
                           batch['abstract']['length'])
            targets = batch['abstract']['words_extended']

            batch_loss = 0
            for i in range(len(logits) - 1):
                batch_loss += criterion(logits[i], targets[:, i+1])
            loss += batch_loss.item() / (targets.size(0) * (targets.size(1) - 1))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (step + 1) % opt['print_every'] == 0:
                print('step ' + str(step+1) + '/' + str(len(data.train_loader)) +': loss '+ str(loss))
                loss = 0
            if (step + 1) % opt['validate_every'] == 0:
                rouge1_sum = 0
                rouge2_sum = 0
                rougeL_sum = 0
                count = 0
                for _, batch in enumerate(data.valid_loader):
                    model.eval()
                    batch = to_device(batch, device=device)
                    preds = model(batch['extracted']['words'],
                                  batch['extracted']['words_extended'],
                                  batch['extracted']['length']).cpu().numpy()
                    golds = batch['abstract']['words_extended'].cpu().numpy()
                    for i in range(len(golds)):
                        pred = strip(preds[i], len(preds[i]), data.vocab.bos_id, data.vocab.eos_id)
                        gold = strip(golds[i], len(golds[i]), data.vocab.bos_id, data.vocab.eos_id)
                        rouge1_sum += rouge_n(pred, gold, n=1)
                        rouge2_sum += rouge_n(pred, gold, n=2)
                        rougeL_sum += rouge_L(pred, gold)
                        count += 1

                print('step ' + str(step+1) + '/' + str(len(data.train_loader)) +
                      ': ROUGE-1 ' + str(rouge1_sum/count) +
                      ' ROUGE-2 ' + str(rouge2_sum/count) +
                      ' ROUGE-L ' + str(rougeL_sum/count))

if __name__ == '__main__':
    opt = set_args()
    data = CnnDm(opt)
    opt['vocab_size'] = len(data.vocab)
    train(opt, data)