#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


import numpy as np

from torch.utils import data
from torchtext.legacy.data import Field, BucketIterator,TabularDataset



################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create


################################################################################
class VectorDataLoader(data.Dataset):

    def __init__(self,file_path,test_mode=False,device="cpu"):
        self.device=device
        self.test_mode=test_mode
        data = np.load(file_path)
        # self.analogy_indices = data["nlg_indices"]
        self.embeddings = data["vectors"]
        self.sentences = data["sentences"]
        assert len(self.embeddings) == len(self.sentences), print("Error: The vectors and sentences are not parallel!")

        self.__length = len(self.sentences)
        self.__vector_size = np.shape(self.embeddings)[-1]

        self.embeddings = torch.from_numpy(self.embeddings).to(device)
        self.embeddings = self.embeddings.float()


    def vector_size(self):
        return self.__vector_size

    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        return self.embeddings[index],self.sentences[index]



def SentenceLoader(path,init_token=None,eos_token=None):
    data_sents=[]
    with open(path,"r+") as file:
        for line in file:
            line = line.strip("\n")
            if init_token and eos_token:
                line =" ".join([init_token] + line.split(" ") + [eos_token])
            else:
                line = line
            data_sents.append(line)

    return data_sents



class TextDataLoader():

    def __init__(self,data_dir=None,train_filename=None,valid_filename=None,test_filename=None,init_token= "SOS",eos_token="EOS",pad_token="PAD",max_length=12,batch_size=128,device="cpu"):
        # special_tokens = {'init_token': "SOS", 'eos_token': "EOS", 'pad_token': "PAD"}
        self.data_dir = data_dir
        self.train_filename= train_filename
        self.valid_filename = valid_filename
        self.test_filename = test_filename
        self.train_path = self.data_dir+self.train_filename
        self.valid_path = self.data_dir + self.valid_filename
        self.test_path = self.data_dir + self.test_filename
        #
        # self.train_path = train_filepath
        # self.valid_path = valid_filepath
        # self.test_path = test_filepath

        self.batch_size = batch_size

        self.init_token=init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length

        self.device= device

        self.corpora_field = Field(tokenize=self.tokenize_en,fix_length=max_length,
                                   init_token=init_token,eos_token=eos_token,pad_token=pad_token,
                                   use_vocab=True,sequential=True)# unk_token="UNK",
        # self.fields = [("src", self.corpora_field), ("trg", self.corpora_field)]
        self.fields = [("sent", self.corpora_field)]


    def tokenize_en(self,text):
        return [w for w in text.split(" ")]


    def generate_vocabulary(self):
        training_set = TabularDataset(path=self.train_path,
                                       format="tsv",
                                       skip_header=False,
                                       fields=self.fields)
        self.corpora_field.build_vocab(training_set, min_freq=1)
        decoder_vocab_size = len(self.corpora_field.vocab)
        print("vocabulary size: {}".format(decoder_vocab_size))

        return self.corpora_field


    def split(self):

        train_data, valid_data, test_data = TabularDataset.splits(path=self.data_dir,
                                                                  format="tsv",
                                                                  train=self.train_filename,
                                                                  test=self.test_filename,
                                                                  validation=self.valid_filename,
                                                                  skip_header=False,
                                                                  fields=self.fields) #path=self.data_dir,

        train_iterator = BucketIterator(train_data,
                                    batch_size=self.batch_size,
                                    device=self.device,
                                    shuffle=True)

        valid_iterator, test_iterator = BucketIterator.splits((valid_data, test_data),
                                                          batch_size=self.batch_size,
                                                          device=self.device,
                                                          sort=False,
                                                          shuffle=False)


        print("train/valid/test size: {}/{}/{}".format(len(train_data),len(valid_data),len(test_data)))

        return train_iterator,valid_iterator,test_iterator









