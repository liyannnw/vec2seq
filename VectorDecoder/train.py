#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from VectorDecoder.networks import basicRNN,ConRNN
from VectorDecoder.networks import init_weights,count_parameters

import torch
import torch.nn as nn
import torch.optim as optim

import math
import time

from VectorDecoder.dataloader import TextDataLoader,SentenceLoader
from VectorDecoder.sent2embedding import embedding
from VectorDecoder.embedding2sent import decoding

from VectorDecoder.pytorchtools import EarlyStopping

import decoder_config as cfg
################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create


################################################################################


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def train(model,embed,train_set,valid_set,valid_path,
            batch_size=128,max_original_sent_len=10,
            corpora_field=None,loss_mode="ml",decoder_net="conrnn",
            epochs=100,clip=1,patience=50,model_save_path=None,
            device="cpu",teacher_forcing_ratio=0.75):



    decode = decoding(embed,model,loss_mode=loss_mode,decoder_net=decoder_net,
                    corpora_field=corpora_field,max_original_sent_len=max_original_sent_len,device=device)

    all_test_sents = SentenceLoader(valid_path, init_token=decode.corpora_field.init_token, eos_token=decode.corpora_field.eos_token)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    optimizer = optim.Adam(decode.decoder.parameters())
    TRG_PAD_IDX = decode.corpora_field.vocab.stoi[decode.corpora_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    if loss_mode == "ml":
        loss2 = nn.MSELoss()
    else:
        loss2 = None


    for epoch in range(epochs):

        start_time1 = time.time()

        # train
        # print("training")
        epoch_loss = 0
        for i, batch in enumerate(train_set):
            truth = batch.sent

            optimizer.zero_grad()

            # convert batch of sentences into context vectors

            batch_sents = [decode.index2sent(truth[:,i],original=False) for i in range(truth.size(1))]

            if decode.encoder.name() == "sbert":
                context_vecs = decode.encoder.sbert_corpus2vecs(batch_sents)
            else:
                context_vecs = [decode.encoder.sent2vec(sent) for sent in batch_sents]
                context_vecs = torch.cat(context_vecs, dim=1)  # ==hidden state == cell state [1,128,300]

            output, output_embeds, input_embeds = decode.dec(context_vecs,sents_ref=truth, train_mode=True,teacher_forcing_ratio=teacher_forcing_ratio)
            # output = [(truth len - 1) * batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            truth = truth[1:].view(-1)#[(truth len - 1) * batch size]

            if loss_mode == "ml":
                loss = criterion(output, truth) + loss2(output_embeds, input_embeds)
            else:
                loss = criterion(output, truth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decode.decoder.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_train_loss = epoch_loss / len(train_set)

        # validate
        # print("validation")
        epoch_loss = 0
        with torch.no_grad():

            for i, batch in enumerate(valid_set):
                truth = batch.sent

                # convert batch of sentences into context vectors
                batch_sents = all_test_sents[i*batch_size:i*batch_size+truth.size(1)]
                # batch_sents = [decode.index2sent(truth[:, i], original=False) for i in range(truth.size(1))]


                if decode.encoder.name() == "sbert":
                    context_vecs = decode.encoder.sbert_corpus2vecs(batch_sents)
                else:

                    context_vecs = [decode.encoder.sent2vec(sent) for sent in batch_sents]
                    context_vecs = torch.cat(context_vecs, dim=1)  # ==hidden state == cell state [1,128,300]

                output, output_embeds, input_embeds = decode.dec(context_vecs, sents_ref=truth, train_mode=False, teacher_forcing_ratio=0)

                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                truth = truth[1:].view(-1)  # [(truth len - 1) * batch size]

                if loss_mode == "ml":
                    loss = criterion(output, truth) + loss2(output_embeds, input_embeds)
                else:
                    loss = criterion(output, truth)

                epoch_loss += loss.item()

        epoch_valid_loss = epoch_loss / len(valid_set)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time1, end_time)

        print('Epoch: {} | Time: {}m {}s'.format(epoch,epoch_mins,epoch_secs))
        print('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(epoch_train_loss,math.exp(epoch_train_loss)))
        print('\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}'.format(epoch_valid_loss,math.exp(epoch_valid_loss)))

        early_stopping(epoch_valid_loss, decode.decoder)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if model_save_path:
        decoder_model_path = model_save_path #+ ".pt"
        torch.save(decode.decoder.state_dict(), decoder_model_path)
        print("saved model.")








