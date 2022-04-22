#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

import math
import time

from vec2seq.dataloader import SentenceLoader
from vec2seq.vector2sentence import decoding

from vec2seq.pytorchtools import EarlyStopping

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create


################################################################################


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



class trainer():
    def __init__(self,decoder=None,encoder=None,train_set=None,valid_set=None,valid_path=None,batch_size=128,max_length=12,
            corpora_field=None,loss_mode="ml",
            epochs=100,clip=1,patience=50,model_save_path=None,
            device="cpu",teacher_forcing_ratio=0.75):

        # self.encoder=encoder  
        # self.decoder = decoder  

        self.train_set= train_set
        self.valid_set = valid_set


        self.epochs = epochs
        self.batch_size = batch_size

        self.clip = clip
        self.model_save_path = model_save_path
        self.loss_mode = loss_mode

        if loss_mode == "ml":
            self.regression_loss = nn.MSELoss()
        else:
            self.regression_loss = None

        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.decode = decoding(encoder=encoder,decoder=decoder,loss_mode=loss_mode,corpora_field=corpora_field,max_length=max_length,device=device)

        self.all_test_sents = SentenceLoader(valid_path, init_token=self.decode.corpora_field.init_token, eos_token=self.decode.corpora_field.eos_token)

        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.optimizer = optim.Adam(self.decode.decoder.parameters())
        self.TRG_PAD_IDX = self.decode.corpora_field.vocab.stoi[self.decode.corpora_field.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.TRG_PAD_IDX)





    def train(self):


        for epoch in range(self.epochs):

            start_time1 = time.time()

            # train
            # print("training")
            epoch_loss = 0
            for i, batch in enumerate(self.train_set):
                truth = batch.sent

                self.optimizer.zero_grad()

                # convert batch of sentences into context vectors

                batch_sents = [self.decode.index2sent(truth[:,i],original=False) for i in range(truth.size(1))]

                # if decode.encoder.name() == "sbert":
                #     context_vecs = decode.encoder.sbert_corpus2vecs(batch_sents)
                # else:
                #     context_vecs = [decode.encoder.sent2vec(sent) for sent in batch_sents]
                #     context_vecs = torch.cat(context_vecs, dim=1)  # ==hidden state == cell state [1,128,300]

                context_vecs = self.decode.encoder.sents2vecs(batch_sents)

                output, output_embeds, input_embeds = self.decode.dec(context_vecs,sents_ref=truth, train_mode=True,teacher_forcing_ratio=self.teacher_forcing_ratio)
                # output = [(truth len - 1) * batch size, output dim]
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                truth = truth[1:].view(-1)#[(truth len - 1) * batch size]

                if self.loss_mode == "ml":
                    loss = self.criterion(output, truth) + self.regression_loss(output_embeds, input_embeds)
                else:
                    loss = self.criterion(output, truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decode.decoder.parameters(), self.clip)
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_train_loss = epoch_loss / len(self.train_set)

            # validate
            # print("validation")
            epoch_loss = 0
            with torch.no_grad():

                for i, batch in enumerate(self.valid_set):
                    truth = batch.sent

                    # convert batch of sentences into context vectors
                    batch_sents = self.all_test_sents[i*self.batch_size:i*self.batch_size+truth.size(1)]
                    # batch_sents = [decode.index2sent(truth[:, i], original=False) for i in range(truth.size(1))]


                    # if decode.encoder.name() == "sbert":
                    #     context_vecs = decode.encoder.sbert_corpus2vecs(batch_sents)
                    # else:
                    #
                    #     context_vecs = [decode.encoder.sent2vec(sent) for sent in batch_sents]
                    #     context_vecs = torch.cat(context_vecs, dim=1)  # ==hidden state == cell state [1,128,300]

                    context_vecs = self.decode.encoder.sents2vecs(batch_sents)

                    output, output_embeds, input_embeds = self.decode.dec(context_vecs, sents_ref=truth, train_mode=False, teacher_forcing_ratio=0)

                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
                    truth = truth[1:].view(-1)  # [(truth len - 1) * batch size]

                    if self.loss_mode == "ml":
                        loss = self.criterion(output, truth) + self.regression_loss(output_embeds, input_embeds)
                    else:
                        loss = self.criterion(output, truth)

                    epoch_loss += loss.item()

            epoch_valid_loss = epoch_loss / len(self.valid_set)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time1, end_time)

            print('Epoch: {} | Time: {}m {}s'.format(epoch,epoch_mins,epoch_secs))
            print('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(epoch_train_loss,math.exp(epoch_train_loss)))
            print('\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}'.format(epoch_valid_loss,math.exp(epoch_valid_loss)))

            self.early_stopping(epoch_valid_loss, self.decode.decoder)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        if self.model_save_path:
            # decoder_model_path = self.model_save_path #+ ".pt"
            torch.save(self.decode.decoder.state_dict(), self.model_save_path)
            print("saved model.")









