#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "26/03/2020", "1.0"  # create

__description__ = "Two sentence embedding methods."

################################################################################
class embedding():
    def __init__(self, method="sbert",mode=None,
                 model_name_or_path='bert-base-nli-mean-tokens', device="cpu"):#mode:{"avg","sum","cls"}
        # self.model = model
        self.tokenizer = None
        self.mode = mode
        self.method = method
        # self.model_path=model_path
        self.device = device

        if self.method == "fasttext":
            self.model = KeyedVectors.load_word2vec_format(model_name_or_path, binary=True)
            self.vector_size = self.model.vector_size

        elif self.method == "bert":
            model_class = BertModel
            tokenizer_class = BertTokenizer
            pretrained_weights = 'bert-base-uncased'

            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
            self.vector_size = self.word2vec("word").size(-1)

        elif self.method == "sbert":
            # self.model = SentenceTransformer('bert-base-nli-mean-tokens')
            self.model = SentenceTransformer(model_name_or_path)
            self.vector_size = 768

        # elif self.method == "rnn":
        #     self.encoder = self.mode
        #     if self.model_path:
        #         self.encoder.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        #     else:
        #         print("==> training encoder...")

    def name(self):
        return self.method

    def word2vec(self,word):
        if self.method == "fasttext":
            vec = self.model[word]

        elif self.method == "bert":

            input_ids = torch.tensor([self.tokenizer.encode(word, add_special_tokens=True)])
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[0]  # [0, 0]
            if self.mode == "cls":
                vec = last_hidden_states[0, 0]
            elif self.mode == "avg":
                vec = torch.mean(last_hidden_states[0, :], dim=0)

            vec = vec.data.numpy()

        elif self.method == "sbert":
            vec = self.model.encode([word])[0]

        # elif self.method == "rnn":
        #     encoder = self.mode
        #     encoder.load_state_dict(torch.load(self.model_path, map_location='cpu'))

        return torch.FloatTensor(vec).view(1,1,-1).to(self.device)


    def sent2vec(self,sentence):
        if self.method == "fasttext":
            wordvectors = []
            for w in sentence.split(" "):
                # wordvectors.append(self.model[word])
                wordvectors.append(self.word2vec(w))
            # wordvectors = torch.cat(wordvectors,dim=1)

            if self.mode == "avg":
                wordvectors = torch.cat(wordvectors, dim=1)
                sentvec = torch.mean(wordvectors,dim=1,keepdim=True)#np.average(wordvectors, axis=0)
            elif self.mode == "sum":
                wordvectors = torch.cat(wordvectors, dim=1)
                sentvec = torch.sum(wordvectors,dim=1,keepdim=True)#np.sum(wordvectors, axis=0)
            elif self.mode == "concat":
                sentvec = torch.cat(wordvectors,dim=2)
            elif self.mode == "multiplication":
                sentvec = torch.mul(wordvectors[0],wordvectors[1])
                for i in range(2,len(wordvectors)):
                    sentvec = torch.mul(sentvec,wordvectors[i])

                return sentvec


        # elif self.method == "bert":
        #     sentvec = self.word2vec(sentence)


        elif self.method == "sbert":
            sentvec = self.word2vec(sentence)



        return sentvec

    def sbert_corpus2vecs(self, corpus):
        output = self.model.encode(corpus)
        output = torch.tensor(output,device=self.device).unsqueeze(0)

        return output




