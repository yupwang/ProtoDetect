import torch
from models.graph import GCGRUModel


class STAE(torch.nn.Module):
    def __init__(self, input_size=2, recon_size=2, st_hidden_size=128, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.st_hidden_size = st_hidden_size
        self.recon_size = recon_size
        self.input_size = input_size
        opt = {
            'num_layer': 2,
            'num_adj': 9,
            'hidden_dim': st_hidden_size,
            'num_feature': input_size
        }
        opt2 = {
            'num_layer': 2,
            'num_adj': 9,
            'hidden_dim': recon_size,
            'num_feature': st_hidden_size
        }
        self.stencoder = GCGRUModel(opt)
        self.stdecoder = GCGRUModel(opt2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(st_hidden_size, int(st_hidden_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(st_hidden_size / 2), hidden_size)
        )
        self.mlp_dec = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, int(st_hidden_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(st_hidden_size / 2), st_hidden_size)
        )

    def encode(self, ts, graph):
        s = ts.shape
        batch_size, time_steps, node_num, feature_size = s[0], s[1], s[2], s[3]
        hidden, _ = self.stencoder(ts, graph)
        stcode = torch.nn.functional.relu(hidden)
        stcode = stcode.reshape([batch_size * node_num, self.st_hidden_size])
        code = self.mlp(stcode)
        code = code.reshape([batch_size, node_num, self.hidden_size])
        return code

    def decode(self, ts, code, graph):
        s = ts.shape
        batch_size, time_steps, node_num, feature_size = s[0], s[1], s[2], s[3]
        code = code.reshape([batch_size * node_num, self.hidden_size])
        stcode = self.mlp_dec(code)
        stcode = stcode.reshape([batch_size, 1, node_num, self.st_hidden_size])
        stcode_series = stcode.repeat(1, time_steps, 1, 1)
        hidden, _ = self.stdecoder(stcode_series, graph, True)
        return hidden

    def forward(self, ts, graph):
        # code: [batch_size, nodes, recon_feat_size]
        code = self.encode(ts, graph)
        if not self.training:
            return code
        else:
            # recon: [B, step, 9, 4]
            recon = self.decode(ts, code, graph)
            return recon


class ProtoDetect(torch.nn.Module):
    def __init__(self, input_nodes=9, input_hidden_size=16, hidden_size=32, proto_num=13, tau_inst=0.1, tau_proto=0.1):
        '''
        Input: batch x nodes x hidden
        '''
        super().__init__()
        self.input_nodes = input_nodes
        self.input_hidden_size = input_hidden_size
        self.input_graph_hidden_size = input_nodes * input_hidden_size
        self.hidden_size = hidden_size
        self.proj_l = torch.nn.Sequential(
            torch.nn.Linear(self.input_graph_hidden_size,
                            int(self.input_graph_hidden_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.input_graph_hidden_size / 2), hidden_size)
        )
        self.proj_g = torch.nn.Sequential(
            torch.nn.Linear(self.input_graph_hidden_size,
                            int(self.input_graph_hidden_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.input_graph_hidden_size / 2), hidden_size)
        )
        self.proj_p = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, proto_num),
            torch.nn.Softmax(1)
        )
        self.c = torch.nn.Parameter(
            torch.randn(proto_num, hidden_size, requires_grad=True).cuda(0))
        self.pred_loss_fnc = torch.nn.MSELoss()
        # temperature
        self.tau_inst = tau_inst
        self.tau_proto = tau_proto

    def forward(self, xl, xg):
        xl = xl.reshape((-1, self.input_graph_hidden_size))
        xg = xg.reshape((-1, self.input_graph_hidden_size))
        zl, zg = self.proj_l(xl), self.proj_g(xg)  # batch_size x hidden_size
        cl, cg = self.proj_p(zl), self.proj_p(zg)  # batch_size x proto_num

        i_info_loss = 0
        c_info_loss = 0
        pred_loss = 0
        if self.training:
            i_info_loss = (_info_nce(zl, zg, self.tau_inst) +
                           _info_nce(zg, zl, self.tau_inst)) / 2
            c_info_loss = (_info_nce(cl.T, cg.T, self.tau_proto) +
                           _info_nce(cg.T, cl.T, self.tau_proto)) / 2
            z = zl + zg / 2
            z_pred = ((cl + cg) / 2) @ self.c
            pred_loss = self.pred_loss_fnc(z_pred, z)
            return zl, zg, i_info_loss, c_info_loss, pred_loss
        return zl, zg, cl, cg, self.c


def _info_nce(query, positive_key, temperature=0.1, reduction='mean'):
    def transpose(x):
        return x.transpose(-2, -1)

    def normalize(*xs):
        return [None if x is None else torch.nn.functional.normalize(x, dim=-1) for x in xs]

    # Normalize to unit vectors
    query, positive_key = normalize(query, positive_key)

    # Cosine between all combinations
    logits = query @ transpose(positive_key)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(query), device=query.device)

    return torch.nn.functional.cross_entropy(logits / temperature, labels, reduction=reduction)
