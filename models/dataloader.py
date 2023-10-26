import torch
from torch_geometric.utils import to_dense_adj, get_laplacian
from torch.utils.data import Dataset
import numpy as np
import json
from torch.utils.data import DataLoader
from models.augmentations import weak_aug, strong_aug


class NYCDataset(Dataset):
    def __init__(self, data, subgraphs, seq_len=12, with_temporal_aug=True):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        s = data.shape
        self.t_len = s[0]
        self.g_len = s[1]

        if with_temporal_aug:
            self.data1, self.data2 = strong_aug(data), weak_aug(data)
        else:
            self.data1, self.data2 = data, data

        # spatial augmentation
        self.subgraphs = subgraphs
        self.subgraphs_adj_form_local = []
        self.subgraphs_adj_form_global = []
        for i in subgraphs:
            edge_index = torch.tensor(i['index']).to(torch.long)
            edge_weight_l = torch.tensor(i['weight_l']).to(torch.float32)
            edge_index_l_norm, edge_weight_l_norm = get_laplacian(
                edge_index, edge_weight_l, normalization='sym')
            adj_norm_l = to_dense_adj(
                edge_index_l_norm, None, edge_weight_l_norm, 9)
            self.subgraphs_adj_form_local.append(adj_norm_l[0])
            edge_weight_g = torch.tensor(i['weight_g']).to(torch.float32)
            edge_index_g_norm, edge_weight_g_norm = get_laplacian(
                edge_index, edge_weight_g, normalization='sym')
            adj_norm_g = to_dense_adj(
                edge_index_g_norm, None, edge_weight_g_norm, 9)
            self.subgraphs_adj_form_global.append(adj_norm_g[0])

    def __getitem__(self, index):
        t_head = index // self.g_len
        t = t_head + self.seq_len
        g = index % self.g_len

        subgraph = self.subgraphs[g]
        sub_nodes = subgraph['related_nodes']
        adj_l = self.subgraphs_adj_form_local[g]
        adj_g = self.subgraphs_adj_form_global[g]

        seq1 = torch.tensor(
            self.data1[(t - self.seq_len):t, sub_nodes]).to(torch.float32)
        seq2 = torch.tensor(
            self.data2[(t - self.seq_len):t, sub_nodes]).to(torch.float32)

        return seq1, seq2, adj_l, adj_g, t, g

    def __len__(self):
        return (self.t_len - self.seq_len) * self.g_len


def get_dataloader(batch_size, with_t_aug):
    data = np.load('dataset/NYC_dynamic_201410_minmax.npy')
    with open('dataset/NYC_graph_dict.json', 'r') as f:
        subgraphs = json.load(f)
    loader = DataLoader(
        NYCDataset(data, subgraphs, with_temporal_aug=with_t_aug),
        batch_size,
        drop_last=True,
        shuffle=True
    )
    return loader
