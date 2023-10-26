import argparse
import torch
import yaml
import numpy as np
from models.dataloader import get_dataloader
from models.ProtoDetect import STAE
from utils.utils import logger, EarlyStopper


def train(model, dataloader, learning_rate=1e-3, num_epochs=6):
    model.train()

    early_stopper = EarlyStopper(patience=10)

    loss_fnc = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

    best_score = None

    for i in range(num_epochs):
        loss_list = []
        for seq1, seq2, adj_l, adj_g, t, g in dataloader:
            seq = seq1.cuda(CUDA_IDX)
            if args.mode == 'local':
                adj = adj_l.cuda(CUDA_IDX)
            else:
                adj = adj_g.cuda(CUDA_IDX)

            seq_hat = model(seq, adj)  # batch x step x node x feature

            loss = loss_fnc(seq_hat, seq)
            loss_list.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        # on epoch end
        scheduler.step()
        avg_loss = np.mean(np.array(loss_list))
        logger(f'{args.mode} - Epoch {i} - Loss: {avg_loss}.')
        if i % 5 == 0:
            torch.save(
                model, f'training_cache/_checkpoints_bone_{args.mode}_{i}.pth')
        if best_score is None or avg_loss < best_score:
            best_score = avg_loss
            torch.save(
                model, f'training_cache/bone_{args.mode}.pth')
        if early_stopper.early_stop(avg_loss):
            logger(f'{args.mode} - Early Stop @ epoch {i}')
            break


if __name__ == '__main__':
    torch.manual_seed(6094)
    np.random.seed(6094)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', help='run in `local` or `global` mode', type=str, default='local')
    args = parser.parse_args()

    with open('config.yaml', 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        hypers = cfg['hypers']

    CUDA_IDX = int(hypers['cuda'])

    data_loader = get_dataloader(hypers['batchSize'], False)
    model = STAE(2, 2)
    model.cuda(CUDA_IDX)
    train(model, data_loader)
