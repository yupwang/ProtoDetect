import argparse
import torch
import yaml
from utils.utils import logger, EarlyStopper
import numpy as np
from models.dataloader import get_dataloader
from models.ProtoDetect import ProtoDetect, STAE

CUDA_IDX = 0

def train(detector, extra_l, extra_g, dataloader, lr=1e-3, num_epochs=6):
    optim = torch.optim.Adam(detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    best_score = None
    early_stopper = EarlyStopper()
    for i in range(num_epochs):
        iloss_list = []
        closs_list = []
        ploss_list = []
        for seq1, seq2, adj_l, adj_g, t, g in dataloader:
            seq1, seq2 = seq1.cuda(CUDA_IDX), seq2.cuda(CUDA_IDX)
            adj_l, adj_g = adj_l.cuda(CUDA_IDX), adj_g.cuda(CUDA_IDX)
            feat_l = extra_l(seq1, adj_l)
            feat_g = extra_g(seq2, adj_g)
            zl, zg, i_info_loss, c_info_loss, pred_loss = detector(
                feat_l, feat_g)
            iloss_list.append(i_info_loss.item())
            closs_list.append(c_info_loss.item())
            ploss_list.append(pred_loss.item())
            optim.zero_grad()
            (i_info_loss + c_info_loss + pred_loss).backward()
            optim.step()

        scheduler.step()
        avg_i = np.mean(np.array(iloss_list))
        avg_c = np.mean(np.array(closs_list))
        avg_p = np.mean(np.array(ploss_list))
        avg_loss = avg_i + avg_c + avg_p
        logger(f'Epoch {i} - i: {avg_i}; c: {avg_c}; p: {avg_p}.')
        if i % 5 == 0:
            torch.save(
                detector, f'training_cache/_checkpoints_proto_detect_{i}.pth')
        if best_score is None or avg_loss < best_score:
            best_score = avg_loss
            torch.save(detector, 'training_cache/proto_detect.pth')
        if early_stopper.early_stop(avg_loss):
            logger(f'Early Stop @ epoch {i}')
            break


def eval(detector, extra_l, extra_g, dataloader):
    with torch.no_grad():
        zl_list = []
        zg_list = []
        cl_list = []
        cg_list = []
        c_proto = None
        for seq1, seq2, adj_l, adj_g, t, g in dataloader:
            seq1, seq2 = seq1.cuda(CUDA_IDX), seq2.cuda(CUDA_IDX)
            adj_l, adj_g = adj_l.cuda(CUDA_IDX), adj_g.cuda(CUDA_IDX)
            feat_l = extra_l(seq1, adj_l)
            feat_g = extra_g(seq2, adj_g)
            zl, zg, cl, cg, proto = detector(
                feat_l, feat_g)
            zl_list.extend(zl.tolist())
            zg_list.extend(zg.tolist())
            cl_list.extend(cl.tolist())
            cg_list.extend(cg.tolist())
            c_proto = proto

        zl, zg, cl, cg = torch.Tensor(zl_list), torch.Tensor(
            zg_list), torch.Tensor(cl_list), torch.Tensor(cg_list)
        proto = c_proto.cpu()

        def compute_score():
            p1 = torch.matmul(cl, proto)
            p2 = torch.matmul(cg, proto)
            dist = torch.nn.MSELoss(reduction='none')
            d1 = dist(zl, p1)
            d2 = dist(zg, p2)
            return (d1 + d2).cpu().numpy()

        score = np.sum(compute_score(), axis=-1)
        np.save(f'anomaly_score.npy', score)
        logger('anomaly score saved')

if __name__ == '__main__':
    torch.manual_seed(6094)
    np.random.seed(6094)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', help='run in `train` or `eval` mode', type=str, default='train')
    args = parser.parse_args()

    with open('config.yaml', 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        hypers = cfg['hypers']

    CUDA_IDX = int(hypers['cuda'])

    extractor_local = torch.load(
        'training_cache/bone_local.pth', map_location=f'cuda:{CUDA_IDX}').eval()
    extractor_global = torch.load(
        'training_cache/bone_global.pth', map_location=f'cuda:{CUDA_IDX}').eval()
    logger('backbone networks loaded')

    dataloader = get_dataloader(hypers['batchSize'], True)
    logger('dataloader built')

    if args.mode == 'train':
        # protodetect
        detector = ProtoDetect(
            hidden_size=hypers['instHiddenSize'], proto_num=hypers['protoNum'],
            tau_inst=hypers['temperatureInst'], tau_proto=hypers['temperatureProto']
        ).cuda(CUDA_IDX).train()
        logger('ProtoDetect inited')

        # train
        train(detector, extractor_local, extractor_global, dataloader)

    if args.mode == 'eval':
        # protodetect model
        detector = torch.load(f'training_cache/proto_detect.pth',
                              map_location=f'cuda:{CUDA_IDX}').eval()
        eval(detector, extractor_local, extractor_global, dataloader)
