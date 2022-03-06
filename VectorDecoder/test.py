#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from VectorDecoder.networks import basicRNN,ConRNN

import torch
import torch.nn as nn

import time

from VectorDecoder.dataloader import TextDataLoader,SentenceLoader
from VectorDecoder.sent2embedding import embedding
from VectorDecoder.embedding2sent import decoding
import csv

from VectorDecoder.eval import eval

import decoder_config as cfg
from VectorDecoder.networks import count_parameters
################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create



################################################################################

def test(model,embed,test_set,test_path,results_save_path=None,
         batch_size=128,max_original_sent_len=10,
        corpora_field=None,loss_mode="ml",decoder_net="conrnn",
        device="cpu"):




    decode = decoding(embed,model,loss_mode=loss_mode,decoder_net=decoder_net,
                    corpora_field=corpora_field,max_original_sent_len=max_original_sent_len,device=device)

    all_test_sents = SentenceLoader(test_path, init_token=decode.corpora_field.init_token, eos_token=decode.corpora_field.eos_token)


    TRG_PAD_IDX = decode.corpora_field.vocab.stoi[decode.corpora_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    if loss_mode == "ml":
        loss2 = nn.MSELoss()
    else:
        loss2 = None

    epoch_loss=0
    decoding_results=[]

    with torch.no_grad():

        for i, batch in enumerate(test_set):
            truth = batch.sent

            batch_sents = all_test_sents[i * batch_size:i * batch_size + truth.size(1)]

            if decode.encoder.name() == "sbert":
                context_vecs = decode.encoder.sbert_corpus2vecs(batch_sents)
            else:

                context_vecs = [decode.encoder.sent2vec(sent) for sent in batch_sents]
                context_vecs = torch.cat(context_vecs, dim=1)  # ==hidden state == cell state [1,128,300]

            results,output,output_embeds,input_embeds = decode.vec2sent(context_vecs,ref=batch_sents)
            decoding_results.extend(results)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            truth = truth[1:].view(-1)  # [(truth len - 1) * batch size]

            if loss_mode == "ml":
                loss = criterion(output, truth) + loss2(output_embeds, input_embeds)
            else:
                loss = criterion(output, truth)

            epoch_loss += loss.item()

        test_loss = epoch_loss / len(test_set)


    if results_save_path:
        results_save_path = results_save_path+".tsv"
        with open(results_save_path,"wt") as file:
            tsv_writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            for line in decoding_results:
                tsv_writer.writerow(line)


    return decoding_results,test_loss






