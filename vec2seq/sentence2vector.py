#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer
import torch

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "26/03/2020", "1.0"  # create

__description__ = "Several sentence embedding methods."

################################################################################
class encoder():
    def __init__(self, model_name=None,model_path=None, device="cpu"):#mode:{"avg","sum","cls"}method="sbert",mode=None,


        self.model_name = model_name
        self.model_path = model_path
        self.device = device


        if 'bert' in self.model_name:
            self.model = SentenceTransformer(self.model_name)
            self.vector_size = self.model.get_sentence_embedding_dimension()
        else:
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
            self.vector_size = self.model.vector_size



    def name(self):
        return self.model_name #method

    def word2vec(self,word):

        if 'bert' in self.model_name:
            vec = self.model.encode([word])[0]
        else:
            vec = self.model[word]




        return torch.FloatTensor(vec).view(1,1,-1).to(self.device)


    def sent2vec(self,sentence):
        if 'bert' in self.model_name:
            sentvec = self.word2vec(sentence)

        else:
            wordvectors = []
            for w in sentence.split(" "):
                # wordvectors.append(self.model[word])
                wordvectors.append(self.word2vec(w))

            if self.model_name == "word-vec-concat":
                sentvec = torch.cat(wordvectors, dim=2)

            elif self.model_name == "word-vec-multiply":
                sentvec = torch.mul(wordvectors[0],wordvectors[1])
                for i in range(2,len(wordvectors)):
                    sentvec = torch.mul(sentvec,wordvectors[i])

            elif self.model_name == "word-vec-avg":
                wordvectors = torch.cat(wordvectors, dim=1)
                sentvec = torch.mean(wordvectors,dim=1,keepdim=True)

            elif self.model_name == "word-vec-sum":
                wordvectors = torch.cat(wordvectors, dim=1)
                sentvec = torch.sum(wordvectors,dim=1,keepdim=True)


        return sentvec




    def sents2vecs(self,sents):
        if 'bert' in self.model_name:
            vecs = self.model.encode(sents)
            vecs = torch.tensor(vecs, device=self.device).unsqueeze(0)

        else:
            vecs = [self.sent2vec(sent) for sent in sents]
            vecs = torch.cat(vecs, dim=1)  # ==hidden state == cell state [1,128,300]

        return vecs







