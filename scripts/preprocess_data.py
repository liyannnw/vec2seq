#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vec2seq.sentence2vector import embedding
import numpy as np
from argparse import ArgumentParser
# from pathlib import Path
import os
import decoder_config as cfg

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"


__usage__='''
python3 preprocess_data.py --embed_model_name_or_path [ENCODER_PATH] --input_path [DATASET_PATH] 
'''

################################################################################
def read_argv():
    parser=ArgumentParser(usage=__usage__)
    parser.add_argument("--embed_method",action="store",default=cfg.embed_method,
                        dest="embed_method")
    parser.add_argument("--embed_mode",action="store",default=cfg.embed_mode,
                        dest="embed_mode")
    parser.add_argument("--embed_model_name_or_path",action="store",default=cfg.embed_model_name_or_path,
                        dest="embed_model_name_or_path")

    parser.add_argument("--text_data_path",action="store",
                        dest="text_data_path")



    return parser.parse_args()



################################################################################
def DataProcessing(file_path=None,embedding_model=None,save_path=None):
    '''
    To encode sentences as vectors,
    and save analogy indices, sentences, vectors into .npz file
    '''

    s2i_dict = dict()
    nlg_indices = list()
    data_size = 0
    embeddings =list()
    sents=list()

    with open(file_path, "r+") as file:
        for line in file:
            data_size += 1
            line = line.strip("\n").split("\t")

            for item in line:
                item_SE = " ".join(["SOS"] + item.split(" ") + ["EOS"])
                vector = embedding_model.sent2vec(item_SE).squeeze().numpy()
                if item not in s2i_dict:
                    s2i_dict[item] = len(s2i_dict)
                    sents.append(item)
                    embeddings.append(vector)


            indexes = [s2i_dict[sent] for sent in line]
            nlg_indices.append(indexes)

    assert data_size == len(nlg_indices)
    # print("{} analogies/{} sentences".format(len(nlg_indices),len(sents)))

    np.savez(save_path,nlg_indices=nlg_indices,sentences=sents,vectors=embeddings)


################################################################################
if __name__ == '__main__':

    opt = read_argv()

    method = opt.embed_method#"fasttext"
    mode = opt.embed_mode#"sum"#"sum"
    model_path = opt.embed_model_name_or_path#"/home/Wang_Liyan/PycharmProjects/SentenceDecoder/model/cc.en.300.init.bin"

    print(">> Loading the embedding model...")
    embed = embedding(method=method, mode=mode, model_name_or_path=model_path, device="cpu")  # encoder_path=None


    print(">> Processing textual analogies and save to a .npz file...\n")
    input_path = opt.text_data_path
    output_path = os.path.splitext(input_path)[0]

    DataProcessing(embedding_model=embed,file_path=input_path,save_path=output_path)

    print(">> Done.")


