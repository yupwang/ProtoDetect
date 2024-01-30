import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, opt, input_size, output_size, activation='sigmoid'):
        super().__init__()
        self.opt = opt
        self.output_size = output_size

        self.fc = nn.Linear(in_features=input_size, out_features=output_size)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x, A):
        """Graph Convolution for batch.
        :param inputs: (B, num_nodes, input_dim)
        :param norm_adj: (B, num_nodes, num_nodes)
        :return
        - Output: `3-D` tensor with shape `(B, num_nodes, rnn_units)`
        """
        x = torch.bmm(A, x)  # (B, num_nodes, input_dim)
        x = self.fc(x)

        return self.activation(x)  # (B, num_nodes, rnn_units)


class GCGRUCell(torch.nn.Module):
    def __init__(self, opt, input_dim, rnn_units):
        super().__init__()

        input_size = input_dim + rnn_units

        self.r_gconv = GCN(opt, input_size=input_size, output_size=rnn_units)
        self.u_gconv = GCN(opt, input_size=input_size, output_size=rnn_units)
        self.c_gconv = GCN(opt, input_size=input_size,
                           output_size=rnn_units, activation='tanh')

    def forward(self, x, h, A):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes, input_dim)
        :param hx: (B, num_nodes, rnn_units)
        :param norm_adj: (B, num_nodes, num_nodes)
        :return
        - Output: A `3-D` tensor with shape `(B, num_nodes, rnn_units)`.
        """
        x_h = torch.cat([x, h], dim=2)

        r = self.r_gconv(x_h, A)
        u = self.u_gconv(x_h, A)

        x_rh = torch.cat([x, r * h], dim=2)

        c = self.c_gconv(x_rh, A)

        h = u * h + (1.0 - u) * c

        return h


class GCGRUModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_rnn_layers = opt['num_layer']

        self.num_nodes = opt['num_adj']
        self.rnn_units = opt['hidden_dim']

        self.dcgru_layers = nn.ModuleList(
            [GCGRUCell(opt=opt, input_dim=opt['num_feature'], rnn_units=self.rnn_units),
             GCGRUCell(opt=opt, input_dim=self.rnn_units, rnn_units=self.rnn_units)])

    def forward(self, inputs, norm_adj, output_series=False):
        """encoder forward pass on t time steps
        :param inputs: shape (batch_size, seq_len, num_node, input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        seq_len = inputs.shape[1]
        s = inputs.shape
        outputs = torch.ones([s[0], s[1], s[2], self.rnn_units], device=inputs.device)
        encoder_hidden_state = None
        for t in range(seq_len):
            output, encoder_hidden_state = self.encoder(
                inputs[:, t, ], norm_adj, encoder_hidden_state)
            outputs[:, t, :, :] = output

        if output_series:
            return outputs, encoder_hidden_state
        else:
            return output, encoder_hidden_state

    def encoder(self, inputs, norm_adj, hidden_state=None):
        """Encoder
        :param inputs: shape (batch_size, self.num_nodes, self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: `2-D` tensor with shape (B, self.num_nodes, self.rnn_units)
                 hidden_state: `2-D` tensor with shape (num_layers, B, self.num_nodes, self.rnn_units)
        """
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.num_nodes, self.rnn_units), device=inputs.device)
        hidden_states = []

        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                output, hidden_state[layer_num,], norm_adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)
