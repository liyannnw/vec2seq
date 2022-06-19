#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

def read_config(path):
    configs = []
    with open(path) as f:
        for line in f:
            configs.append(json.loads(line.strip()))
    return configs[0]


def save_vocab(vocab, path):
    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')






class read_vocab():
    def __init__(self,path):
        # self.vocab = vocab(path)
        self.stoi = dict()

        with open(path, 'r') as f:
            for line in f:
                index, token = line.strip().split('\t')
                self.stoi[token] = int(index)

        self.reversed = self.reverse()
        self.itos = self.reversed

    def reverse(self):
        return  {v: k for k, v in self.stoi.items()}

    def vocab(self):
        return self.stoi