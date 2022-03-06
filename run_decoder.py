#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from VectorDecoder.networks import basicRNN,ConRNN

import torch


from VectorDecoder.dataloader import TextDataLoader
from VectorDecoder.sent2embedding import embedding


from os import path
import sys
from VectorDecoder.eval import eval

import decoder_config as cfg
import argparse

from VectorDecoder.networks import init_weights,count_parameters

from VectorDecoder.train import train
from VectorDecoder.test import test
################################################################################

__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "30/07/2020", "1.0"# Create
__description__ = "Training or test the decoder for sentence vectors."


################################################################################


def read_argv():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run", action="store",default=cfg.run,
                        help="train or test")

    ###################################################
    # data
    parser.add_argument("--data_dir", dest='data_dir', action="store",default=cfg.data_dir,
                        help="")
    parser.add_argument("--train_filename",dest='train_filename',action="store",default=cfg.train_filename,
                        help="")
    parser.add_argument("--valid_filename", dest='valid_filename', action="store",default=cfg.valid_filename,
                        help="")
    parser.add_argument("--test_filename", dest='test_filename', action="store",default=cfg.test_filename,
                        help="")

    parser.add_argument("--start_of_speech", dest='start_of_speech', action="store",default=cfg.start_of_speech,
                        help="")
    parser.add_argument("--end_of_speech", dest='end_of_speech', action="store",default=cfg.end_of_speech,
                        help="")
    parser.add_argument("--padding_token", dest='padding_token', action="store",default=cfg.padding_token,
                        help="")
    parser.add_argument("--max_original_sent_len", dest='max_original_sent_len', action="store",default=cfg.max_original_sent_len,
                        help="")
    parser.add_argument("--fixed_length", dest='fixed_length', action="store",default=cfg.fixed_length,
                        help="")

    ###################################################
    # configuration of decoder
    parser.add_argument("--BATCH_SIZE", dest='BATCH_SIZE', action="store",default=cfg.BATCH_SIZE,
                        help="The file path of source embedding model.")
    parser.add_argument("--N_EPOCHS", dest='N_EPOCHS', action="store",default=cfg.N_EPOCHS,
                        help="")
    parser.add_argument("--CLIP", dest='CLIP', action="store",default=cfg.CLIP,
                        help="")
    parser.add_argument("--PATIENCE", dest='PATIENCE', action="store",default=cfg.PATIENCE,
                        help="")
    parser.add_argument("--N_LAYERS", dest='N_LAYERS', action="store", default=cfg.N_LAYERS,
                        help="")
    parser.add_argument("--DEC_DROPOUT", dest='DEC_DROPOUT', action="store", default=cfg.DEC_DROPOUT,
                        help="")
    parser.add_argument("--teacher_forcing_ratio", dest='teacher_forcing_ratio', action="store", default=cfg.teacher_forcing_ratio,
                        help="")

    parser.add_argument("--decoder_loss", dest='decoder_loss', action="store",default=cfg.decoder_loss,
                        help="")
    parser.add_argument("--decoder_net", dest='decoder_net', action="store",default=cfg.decoder_net,
                        help="")

    parser.add_argument("--pretrained_decoder_path", dest='pretrained_decoder_path', action="store",default=cfg.pretrained_decoder_path,
                        help="")

    ###################################################
    # sentence embedding
    parser.add_argument("--embed_method", dest='embed_method', action="store",default=cfg.embed_method,
                        help="")
    parser.add_argument("--embed_mode", dest='embed_mode', action="store",default=cfg.embed_mode,
                        help="")
    parser.add_argument("--embed_model_name_or_path", dest='embed_model_name_or_path', action="store",default=cfg.embed_model_name_or_path,
                        help="")


    ###################################################
    # save
    parser.add_argument("--result_dir", dest='result_dir', action="store",default=cfg.result_dir,
                        help="")
    parser.add_argument("--save_path", dest='save_path', action="store",default=cfg.save_path,
                        help="save decoder model or decoding results")


    return parser.parse_args()



################################################################################
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()


    ###################################################
    # LOADING PRE-TRAINED EMBEDDING MODEL
    print("===LOADING PRE-TRAINED EMBEDDING MODEL===\n")
    embed = embedding(method=opt.embed_method, mode=opt.embed_mode, model_name_or_path=opt.embed_model_name_or_path, device=device)


    ###################################################
    # LOADING DATASETS AND CREATING VOCABULARY FOR DECODER
    print("===LOADING DATASETS AND CREATING VOCABULARY FOR DECODER===\n")
    DL = TextDataLoader(opt.data_dir,opt.train_filename,opt.valid_filename,opt.test_filename,
                    init_token=opt.start_of_speech,eos_token=opt.end_of_speech,pad_token=opt.padding_token,
                    batch_size=opt.BATCH_SIZE,fix_length=opt.fixed_length,device=device)

    vocab = DL.generate_vocabulary()
    train_set,valid_set,test_set = DL.datasets()


    ###################################################
    # ESTABLISHING DECODER MODEL
    print("===ESTABLISHING DECODER MODEL===\n")
    OUTPUT_DIM = len(vocab.vocab)
    DEC_EMB_DIM = embed.vector_size  # 256
    HID_DIM = embed.vector_size  # 512
    # N_LAYERS = opt.N_LAYERS#1
    # DEC_DROPOUT = opt.DEC_DROPOUT#0.4
    #
    # teacher_forcing_ratio= cfg.teacher_forcing_ratio#0.75

    if opt.decoder_net == "conrnn":
        dec = ConRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, opt.N_LAYERS, opt.DEC_DROPOUT,loss=opt.decoder_loss)
    elif opt.decoder_net == "basicrnn":
        dec = basicRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, opt.N_LAYERS, opt.DEC_DROPOUT, loss=opt.decoder_loss)

    dec = dec.to(device)
    print(dec)
    print('The model has {} trainable parameters'.format(count_parameters(dec)))


    ###################################################
    # TRAINING OR TESTING THE DECODING NETWORK
    run = opt.run
    print("==={}ing THE DECODING NETWORK===\n".format(run))
    if run == "train":
        if path.exists(opt.save_path+".pt"):
            print("Error: The file already exists, please change the file name.")
            sys.exit(0)

        dec.apply(init_weights)

        valid_path = opt.data_dir + opt.valid_filename

        train(dec, embed, train_set, valid_set, valid_path,
              batch_size=opt.BATCH_SIZE, corpora_field=vocab,
              loss_mode=opt.decoder_loss, decoder_net=opt.decoder_net,
              epochs=opt.N_EPOCHS, clip=opt.CLIP, patience=opt.PATIENCE,
              max_original_sent_len=opt.max_original_sent_len, model_save_path=opt.save_path,
              device=device, teacher_forcing_ratio=opt.teacher_forcing_ratio)

    elif run == "test":
        if path.exists(opt.save_path+".tsv"):
            print("Error: The file already exists, please change the file name.")
            sys.exit(0)

        dec.load_state_dict(torch.load(opt.pretrained_decoder_path))

        test_path = opt.data_dir + opt.test_filename
        results_save_path = opt.save_path

        results, test_loss = test(dec, embed, test_set, test_path, results_save_path=results_save_path,
                                  batch_size=opt.BATCH_SIZE, max_original_sent_len=opt.max_original_sent_len,
                                  corpora_field=vocab, loss_mode=opt.decoder_loss, decoder_net=opt.decoder_net,
                                  device=device)
        eval(results)
