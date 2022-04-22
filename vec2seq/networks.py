#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create

__description__ = "The architecture of basic RNN and ConRNN"

################################################################################

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class basicRNN(nn.Module):
    def __init__(self, vocab_size=None, vector_dim=None,hid_dim=None,  n_layers=None, dropout=0.4,loss="ml"):
        super().__init__()

        self.loss = loss
        self.emb_dim = vector_dim
        self.output_dim = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim,self.n_layers, dropout=dropout)
        self.fc_out = nn.Linear(self.hid_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

        self.net_type = 'basicrnn'



    def forward(self, input, hidden, cell):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]


        if self.loss == "sl":
            return prediction, hidden, cell
        elif self.loss == "ml":
            return output, prediction, hidden, cell





class ConRNN(nn.Module):
    def __init__(self, vocab_size=None, vector_dim=None, hid_dim=None, n_layers=None, dropout=0.4, loss="ml"):
        super().__init__()

        self.loss = loss
        self.emb_dim = vector_dim
        self.output_dim = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers


        self.rnn = nn.LSTM(self.emb_dim+self.hid_dim, self.hid_dim, self.n_layers, dropout=dropout)
        self.fc_out = nn.Linear(self.emb_dim+ self.hid_dim*2, self.output_dim)

        self.dropout = nn.Dropout(dropout)

        self.net_type = 'conrnn'



    def forward(self, input, hidden, cell,context):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        input_con = torch.cat((input,context),dim=2)
        # input_con = [batch size]


        output, (hidden, cell) = self.rnn(input_con, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]

        output_con = torch.cat((input.squeeze(0), output.squeeze(0), context.squeeze(0)), dim=1)
        # output = [seq len, batch size, hid dim * n directions]

        prediction = self.fc_out(output_con)
        # prediction = [batch size, output dim]


        if self.loss == "sl":
            return prediction, hidden, cell
        elif self.loss == "ml":
            return output, prediction, hidden, cell


################################################################################
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # configs
    OUTPUT_DIM = 10380
    DEC_EMB_DIM = 768
    HID_DIM = 768
    N_LAYERS = 1
    DEC_DROPOUT = 0.4
    loss_mode="ml"

    decoder = ConRNN(vocab_size=OUTPUT_DIM, vector_dim=DEC_EMB_DIM, n_layers= N_LAYERS, dropout=DEC_DROPOUT,loss=loss_mode)
    print(decoder)