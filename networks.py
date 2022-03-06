#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"



################################################################################


class LinearFCN(nn.Module):#fcNetwork
    def __init__(self,layers_size=[512,512,512],
                 vector_size=300,
                 output_n=1,
                 mode="regression",
                 compose_mode="concat"):

        super(LinearFCN, self).__init__()
        # self.input_n = input_n
        self.output_n = output_n
        self.vector_size = vector_size
        self.mode = mode
        self.compose_mode = compose_mode

        if self.compose_mode == "concat":
            self.input_n = 3
        else:
            self.input_n = 1



        self.input_size = self.input_n * self.vector_size
        self.output_size = self.output_n * self.vector_size

        if self.mode == "regression":
            layers_size.append(self.output_size)
        elif self.mode == "classification":
            layers_size.append(self.output_n)
        self.layers_size = [self.input_size]
        self.layers_size.extend(layers_size)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.layers = nn.ModuleList()
        for i in range(len(self.layers_size)-1):
            self.layers.append(nn.Linear(self.layers_size[i],self.layers_size[i+1]))
            # self.layers(self.relu)
            # self.relu(self.layers)

    def TripleComposition(self,input, compose_mode="concat"):

        n_input = input.size(1)
        vector_size = input.size(2)

        if compose_mode == "concat":
            return input.view(-1, n_input * vector_size)

        elif compose_mode == "sum":
            return torch.sum(input, dim=1)

        elif compose_mode == "ap":
            new_input = []
            for item in input:
                A, B, C = item
                tmp = B - A + C
                tmp = tmp.unsqueeze(0)
                new_input.append(tmp)
            new_input = torch.cat(new_input, dim=0)
            return new_input

    def forward(self,batch):
        # print("input_size",self.input_size)
        # print(batch.size())
        batch = self.TripleComposition(batch,compose_mode=self.compose_mode)

        batch = batch.view(-1,self.input_size)

        for i in range(len(self.layers)-1):
            # batch = F.tanh(self.layers[i](batch))
            batch = self.relu(self.layers[i](batch))


        if self.mode == "regression":
            batch = self.relu(self.layers[-1](batch))#batch = F.tanh(self.layers[-1](batch))  ###
            # batch = torch.relu(self.layers[-1](batch))
            # batch = F.tanh(self.layers[-1](batch))

            # batch = self.dropout(self.layers[-1](batch))
            # batch=batch.view(-1,self.output_n,self.vector_size)
        elif self.mode == "classification":
            batch = F.log_softmax(self.layers[-1](batch))
            batch = batch.view(-1, self.output_n)

        return batch