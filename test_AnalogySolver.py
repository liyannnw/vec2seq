#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils import data
import csv
from argparse import ArgumentParser

from networks import LinearFCN
from train_AnalogySolverinSpace import dataFile_loader

from vec2seq.sentence2vector import embedding
from vec2seq.dataloader import TextDataLoader
from vec2seq.networks import ConRNN,basicRNN

from vec2seq.embedding2sent import decoding

import decoder_config as dec_cfg
import solver_config as as_cfg

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"

__usage__='''
python3 test_AnalogySolver,py --data_dir_path [DIR_PATH TO TEST SET] --solver_path [SOLVER_PATH] --results_save_path [SAVE_PATH]
'''

################################################################################

def vocab_of_decoder(decoder_cfg,device="cpu"):
    DL = TextDataLoader(decoder_cfg.data_dir,decoder_cfg.train_filename,decoder_cfg.valid_filename,decoder_cfg.test_filename,
                    init_token=decoder_cfg.start_of_speech,eos_token=decoder_cfg.end_of_speech,pad_token=decoder_cfg.padding_token,
                    batch_size=decoder_cfg.BATCH_SIZE,fix_length=decoder_cfg.fixed_length,device=device)

    vocab = DL.generate_vocabulary()
    return vocab



def decoder_loader(decoder_cfg,vector_size=None,device="cpu"):

    DL = TextDataLoader(decoder_cfg.data_dir,decoder_cfg.train_filename,decoder_cfg.valid_filename,decoder_cfg.test_filename,
                    init_token=decoder_cfg.start_of_speech,eos_token=decoder_cfg.end_of_speech,pad_token=decoder_cfg.padding_token,
                    batch_size=decoder_cfg.BATCH_SIZE,fix_length=decoder_cfg.fixed_length,device=device)

    decoder_vocab = DL.generate_vocabulary()

    # parameters of decoder
    # decoder_vocab=vocab_of_decoder(dec_cfg)
    OUTPUT_DIM = len(decoder_vocab.vocab)
    DEC_EMB_DIM = vector_size
    HID_DIM = vector_size
    N_LAYERS = dec_cfg.N_LAYERS
    DEC_DROPOUT = dec_cfg.DEC_DROPOUT
    teacher_forcing_ratio= dec_cfg.teacher_forcing_ratio#0.75
    decoder_net = dec_cfg.decoder_net

    if decoder_net == "conrnn":
        decoder = ConRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,loss=dec_cfg.decoder_loss)
    elif decoder_net == "basicrnn":
        decoder = basicRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, loss=dec_cfg.decoder_loss)


    decoder_model_path= dec_cfg.save_path
    decoder.load_state_dict(torch.load(decoder_model_path))
    decoder = decoder.to(device)

    return decoder,decoder_vocab



def test(test_set,model,decoder,embed,batch_size=32,dec_config=None,decoder_vocab=None,device="cpu"):

    test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()
    decoder.eval()

    if dec_config:
        max_sent_len= dec_config.max_original_sent_len
        decoder_loss= dec_config.decoder_loss
        decoder_net= dec_config.decoder_net

    decode = decoding(embed,decoder,
                      loss_mode=decoder_loss,decoder_net=decoder_net,
                      corpora_field=decoder_vocab,
                      max_original_sent_len=max_sent_len,device=device)

    nlg_results = []
    with torch.no_grad():
        for index, (batch, truth, sents) in enumerate(test_generator):
            output = model(batch)
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
    parser.add_argument("--data_dir_path",action="store",default=as_cfg.data_dir_path,
                        dest="data_dir_path",help="...")

    parser.add_argument("--batch_size",action="store",default=as_cfg.batch_size,type=int,
                        dest="batch_size",help="...")
    parser.add_argument("--solver_path",action="store",default=as_cfg.model_save_path,
                        dest="solver_path",help="...")
    parser.add_argument("--compose_mode",action="store",default=as_cfg.compose_mode,
                        dest="compose_mode",help="...")
    parser.add_argument("--nb_output_item",action="store",default=as_cfg.nb_output_item,type=int,
                        dest="nb_output_item",help="...")
    parser.add_argument("--layers_size",action="store",default=as_cfg.layers_size,
                        dest="layers_size",nargs='+', type=int,help="...")


    parser.add_argument("--results_save_path",action="store",default=as_cfg.results_save_path,
                        dest="results_save_path",help="...")


    return parser.parse_args()

################################################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()

    print(">> Loading test data...")
    data_dir_path = opt.data_dir_path
    test_set = dataFile_loader(data_dir_path,data="test",device=device)


    print(">> Loading models:")
    print(">>>> encoder...")
    encoder = embedding(method=dec_cfg.embed_method, mode=dec_cfg.embed_mode,
                      model_name_or_path=dec_cfg.embed_model_name_or_path, device=device)


    print(">>>> decoder...")
    decoder,decoder_vocab = decoder_loader(dec_cfg,vector_size=encoder.vector_size,device=device)


    print(">>>> analogy solver in vector space...")
    solver = LinearFCN(vector_size=encoder.vector_size,layers_size=opt.layers_size,
                       output_n=opt.nb_output_item,compose_mode=opt.compose_mode)
    solver.load_state_dict(torch.load(opt.solver_path))
    solver = solver.to(device)


    results = test(test_set, solver, decoder, encoder,
                   batch_size=opt.batch_size, dec_config=dec_cfg,
                   decoder_vocab=decoder_vocab,device=device)

    if opt.results_save_path:
        with open(opt.results_save_path,"wt") as file:
            tsv_writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow(["#A","#B","#C","#D_ref","#D_hyp"])
            for line in results:
                tsv_writer.writerow(line)
        print("saved!")









