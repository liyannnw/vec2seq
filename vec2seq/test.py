#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from vec2seq.vector2sentence import decoding
import csv

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create



################################################################################

def test(encoder=None,decoder=None,test_set=None,results_save_path=None,decoder_cfg=None,
         corpora_field=None,device="cpu",added_special_token=False):


    decode = decoding(encoder=encoder, decoder=decoder, loss_mode=decoder_cfg["decoder_loss"], corpora_field=corpora_field,
                           max_length=decoder_cfg["max_length"], device=device, init_token=decoder_cfg["init_token"], eos_token=decoder_cfg["eos_token"],
                           pad_token=decoder_cfg["pad_token"])


    # TRG_PAD_IDX = decode.corpora_field.stoi[decoder_cfg["pad_token"]]


    decoding_results=[]

    with torch.no_grad():
        for i, sent in enumerate(test_set):

            context_vecs = decode.encoder.sents2vecs([sent])
            results,output,output_embeds,input_embeds = decode.vec2sent(context_vecs,ref=[sent],added_special_token=added_special_token)


            decoding_results.extend(results)




    if results_save_path:
        # results_save_path = results_save_path+".tsv"
        with open(results_save_path,"wt") as file:
            tsv_writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            for line in decoding_results:
                tsv_writer.writerow(line)


    return decoding_results #,test_loss






