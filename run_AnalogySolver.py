#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import torch
from torch.utils import data
import csv
from argparse import ArgumentParser

from networks import LinearFCN
import numpy as np

from vec2seq.sentence2vector import encoding
from vec2seq.dataloader import TextDataLoader
from vec2seq.networks import ConRNN,basicRNN

from vec2seq.vector2sentence import decoding

from utils import AnalogicalDataLoader
from vec2seq.utils import read_config
from vec2seq.utils import read_vocab

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"

__usage__='''

'''

################################################################################

def SentenceAnalogySolver(test_set=None,vecsolver=None,decoder=None,vocab=None,batch_size=1,decoder_cfg=None,device="cpu"):

    vecsolver.eval()
    decoder.eval()

    test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)



    encoder = encoding(model_name=decoder_cfg["encoder_name"], model_path=decoder_cfg["encoder_path"], device=device)

    decode = decoding(encoder=encoder, decoder=decoder, loss_mode=decoder_cfg["decoder_loss"], corpora_field=vocab,
                           max_length=decoder_cfg["max_length"], device=device, init_token=decoder_cfg["init_token"], eos_token=decoder_cfg["eos_token"],
                           pad_token=decoder_cfg["pad_token"])





    nlg_results = []
    with torch.no_grad():
        for index, (batch, truth, sents) in enumerate(test_generator):
            output = vecsolver(batch)
            context_vecs = output.unsqueeze(0)

            ref_sents = sents[-1]
            nlg_sents = list(map(list, zip(*sents)))

            results, output, output_embeds, input_embeds = decode.vec2sent(context_vecs,
                                                                           ref=ref_sents,
                                                                           added_special_token=False)

            for n,nlg in enumerate(nlg_sents):
                nlg_results.append(nlg+[results[n][1]])


    return nlg_results



################################################################################

def read_argv():
    parser=ArgumentParser(usage=__usage__)


    parser.add_argument("--decoder_path", dest='decoder_path', action="store", default=None)
    parser.add_argument("--solver_path", dest='solver_path', action="store", default=None)

    parser.add_argument("--test_filepath", dest='test_filepath', action="store", default=None)

    parser.add_argument("--save_filepath",action="store",default=None,
                        dest="save_filepath",help="...")


    return parser.parse_args()

################################################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()

    print(">> Loading test data...")
    test_set = AnalogicalDataLoader(opt.test_filepath, test_mode=True, device=device)

    print(">> Loading models:")
    decoder_cfgs = read_config(opt.decoder_path + '/config.json')


    print(">>>> decoder...")

    vocab = read_vocab(opt.decoder_path+'/vocab.txt')


    net=ConRNN if decoder_cfgs["decoder_net"] == "conrnn" else basicRNN

    decoder = net(vocab_size=decoder_cfgs["vocab_size"], vector_dim=decoder_cfgs["vector_dim"], hid_dim=decoder_cfgs["hid_dim"],
                  n_layers=decoder_cfgs["n_layers"], loss=decoder_cfgs["decoder_loss"])
    decoder.load_state_dict(torch.load(opt.decoder_path + '/checkpoint.pt',map_location='cuda:0'))
    decoder = decoder.to(device)



    print(">>>> analogy solver in vector space...")
    solver_cfgs = read_config(opt.solver_path+'/config.json')



    solver = LinearFCN(vector_size=solver_cfgs["vector_size"],layers_size=solver_cfgs["layers_size"],
                       output_n=solver_cfgs["output_n"],compose_mode=solver_cfgs["compose_mode"])
    solver.load_state_dict(torch.load(opt.solver_path+'/checkpoint.pt',map_location='cuda:0'))
    solver = solver.to(device)


    results = SentenceAnalogySolver(test_set=test_set,vecsolver=solver,decoder=decoder,vocab=vocab,decoder_cfg=decoder_cfgs,device=device)

    acc=0
    for line in results:
        if line[3] == line[4]:
            acc+=1
    print("Acc: {}%".format(np.round(100*acc/len(results),2)))


    if opt.save_filepath:
        with open(opt.save_filepath,"w") as file:
            tsv_writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow(["#A","#B","#C","#D_ref","#D_hyp"])
            for line in results:
                tsv_writer.writerow(line)
        print("saved!")









