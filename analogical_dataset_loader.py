#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import numpy as np

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"

################################################################################


class AnalogicalDataLoader(data.Dataset):

    def __init__(self,file_path,test_mode=False,device="cpu"):
        self.device=device
        self.test_mode=test_mode
        # self.embed=embed

        data = np.load(file_path)
        self.analogy_indices = data["nlg_indices"]
        self.embeddings = data["vectors"]
        self.sentences = data["sentences"]
        self.__length = len(self.analogy_indices)
        self.__vector_size = np.shape(self.embeddings)[-1]

        self.embeddings = torch.from_numpy(self.embeddings).to(device)
        self.embeddings = self.embeddings.float()


    def vector_size(self):
        return self.__vector_size
    def __len__(self):
        return self.__length


    def __getitem__(self, index):
        analogy = self.analogy_indices[index]
        if self.test_mode:
            return self.embeddings[analogy[:3]], self.embeddings[analogy[-1]],self.sentences[analogy[:]].tolist()
        else:
            return self.embeddings[analogy[:3]], self.embeddings[analogy[-1]]

