#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vec2seq.networks import basicRNN,ConRNN

import torch


from vec2seq.dataloader import TextDataLoader
from vec2seq.sentence2vector import encoder


from os import path
import sys
from vec2seq.eval import eval

import decoder_config as cfg
import argparse

from vec2seq.networks import init_weights,count_parameters

from vec2seq.train import trainer
from vec2seq.test import test

from tabulate import tabulate
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
    # parser.add_argument("--max_original_sent_len", dest='max_original_sent_len', action="store",default=cfg.max_original_sent_len,
    #                     help="")
    parser.add_argument("--max_length", dest='max_length', action="store",default=cfg.max_length,
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
    # parser.add_argument("--embed_method", dest='embed_method', action="store",default=cfg.embed_method,
    #                     help="")
    # parser.add_argument("--embed_mode", dest='embed_mode', action="store",default=cfg.embed_mode,
    #                     help="")
    parser.add_argument("--encoder_name", dest='encoder_name', action="store",default=cfg.encoder_name,
                        help="")
    parser.add_argument("--encoder_path", dest='encoder_path', action="store",default=cfg.encoder_path,
                        help="")


    ###################################################
    # save
    parser.add_argument("--result_dir", dest='result_dir', action="store",default=cfg.result_dir,
                        help="")
    parser.add_argument("--model_save_path", dest='model_save_path', action="store",default=cfg.model_save_path,
                        help="path to decoder model")
    parser.add_argument("--sentence_save_path", dest='sentence_save_path', action="store",default=cfg.sentence_save_path,
                        help="path to generated sentences")


    return parser.parse_args()



################################################################################
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()


    ###################################################
    # LOADING PRE-TRAINED EMBEDDING MODEL
    print("===LOADING PRE-TRAINED EMBEDDING MODEL===\n")
    # embed = embedding(method=opt.embed_method, mode=opt.embed_mode, model_name_or_path=opt.embed_model_name_or_path, device=device)
    encoder = encoder(model_name=opt.encoder_name,model_path=opt.encoder_path, device=device)


    ###################################################
    # LOADING DATASETS AND CREATING VOCABULARY FOR DECODER
    print("===LOADING DATASETS AND CREATING VOCABULARY FOR DECODER===\n")
    dataset = TextDataLoader(data_dir=opt.data_dir,
                             train_filename=opt.train_filename,
                             valid_filename=opt.valid_filename,
                             test_filename=opt.test_filename,
                             init_token=opt.start_of_speech,
                             eos_token=opt.end_of_speech,
                             pad_token=opt.padding_token,
                             batch_size=opt.BATCH_SIZE,
                             max_length=opt.max_length,
                             device=device)

    # dataset = TextDataLoader(train_filepath=opt.train_filename,
    #                          valid_filepath=opt.valid_filename,
    #                          test_filepath=opt.test_filename,
    #                          init_token=opt.start_of_speech,
    #                          eos_token=opt.end_of_speech,
    #                          pad_token=opt.padding_token,
    #                          batch_size=opt.BATCH_SIZE,
    #                          fix_length=opt.fixed_length,
    #                          device=device)

    vocab = dataset.generate_vocabulary()
    train_set,valid_set,test_set = dataset.split()

    

    ###################################################
    # ESTABLISHING DECODER MODEL
    print("===ESTABLISHING DECODER MODEL===\n")
    OUTPUT_DIM = len(vocab.vocab)
    DEC_EMB_DIM = encoder.vector_size  # 256
    HID_DIM = encoder.vector_size  # 512
    # N_LAYERS = opt.N_LAYERS#1
    # DEC_DROPOUT = opt.DEC_DROPOUT#0.4
    #
    # teacher_forcing_ratio= cfg.teacher_forcing_ratio#0.75

    if opt.decoder_net == "conrnn":
        decoder = ConRNN(vocab_size=OUTPUT_DIM , vector_dim=DEC_EMB_DIM, hid_dim=HID_DIM, n_layers=opt.N_LAYERS, loss=opt.decoder_loss)
    elif opt.decoder_net == "basicrnn":
        decoder = basicRNN(vocab_size=OUTPUT_DIM , vector_dim=DEC_EMB_DIM, hid_dim=HID_DIM, n_layers=opt.N_LAYERS, loss=opt.decoder_loss)

    decoder = decoder.to(device)
    print(decoder)
    print('The model has {} trainable parameters'.format(count_parameters(decoder)))


    ###################################################
    # TRAINING OR TESTING THE DECODING NETWORK
    run = opt.run
    print("==={}ing THE DECODING NETWORK===\n".format(run))
    if run == "train":
        if path.exists(opt.model_save_path): #+".pt"
            print("Error: duplicate name of the path for saving model.")
            sys.exit(0)

        decoder.apply(init_weights)

        valid_path = opt.data_dir + opt.valid_filename

        trainer = trainer(decoder=decoder,encoder=encoder,train_set=train_set,valid_set=valid_set,valid_path=valid_path,batch_size=opt.BATCH_SIZE,max_length=opt.max_length,
            corpora_field=vocab,loss_mode=opt.decoder_loss,
            epochs=10,clip=opt.CLIP,patience=opt.PATIENCE,model_save_path=opt.model_save_path,
            device=device,teacher_forcing_ratio=opt.teacher_forcing_ratio)

        trainer.train()

        # train(decoder, encoder, train_set, valid_set, valid_path,
        #       batch_size=opt.BATCH_SIZE, corpora_field=vocab,
        #       loss_mode=opt.decoder_loss, decoder_net=opt.decoder_net,
        #       epochs=opt.N_EPOCHS, clip=opt.CLIP, patience=opt.PATIENCE,
        #       max_original_sent_len=opt.max_original_sent_len, model_save_path=opt.save_path,
        #       device=device, teacher_forcing_ratio=opt.teacher_forcing_ratio)

    elif run == "test":
        # if path.exists(opt.save_path):
        #     print("Error: duplicate name of the path for saving results.")
        #     sys.exit(0)

        decoder.load_state_dict(torch.load(opt.pretrained_decoder_path))

        test_path = opt.data_dir + opt.test_filename
        # results_save_path = opt.save_path

        results,test_loss = test(encoder=encoder,decoder=decoder,
            test_set=test_set,test_path=test_path,results_save_path=opt.sentence_save_path,
            batch_size=opt.BATCH_SIZE,corpora_field=vocab,loss_mode=opt.decoder_loss,device=device)

        # results, test_loss = test(dec, encoder, test_set, test_path, results_save_path=results_save_path,
        #                           batch_size=opt.BATCH_SIZE, max_original_sent_len=opt.max_original_sent_len,
        #                           corpora_field=vocab, loss_mode=opt.decoder_loss, decoder_net=opt.decoder_net,
        #                           device=device)
        scores = eval(results)

        

        print(tabulate(results, headers=["Reference","Generation"]))
        

        print(tabulate(scores.items()))


