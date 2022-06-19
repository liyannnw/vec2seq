#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vec2seq.networks import basicRNN,ConRNN

import torch
import json

from vec2seq.dataloader import TextDataLoader
from vec2seq.sentence2vector import encoding

import os


from vec2seq.eval import eval


import argparse

from vec2seq.networks import init_weights,count_parameters

from vec2seq.train import trainer
from vec2seq.test import test

from tabulate import tabulate

from vec2seq.utils import read_config
################################################################################

__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "30/07/2020", "1.0"# Create
__description__ = "Training or test the decoder for sentence vectors."


################################################################################


def read_argv():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="run", action="store",default='train',
                        help="train or test")

    ###################################################
    # data
    parser.add_argument("--data_dir", dest='data_dir', action="store",default=None,
                        help="")
    parser.add_argument("--train_filename",dest='train_filename',action="store",default=None,
                        help="")
    parser.add_argument("--valid_filename", dest='valid_filename', action="store",default=None,
                        help="")
    parser.add_argument("--test_filename", dest='test_filename', action="store",default=None,
                        help="")

    parser.add_argument("--init_token", dest='init_token', action="store",default="SOS",
                        help="")
    parser.add_argument("--eos_token", dest='eos_token', action="store",default="EOS",
                        help="")
    parser.add_argument("--pad_token", dest='pad_token', action="store",default="PAD",
                        help="")
    # parser.add_argument("--max_original_sent_len", dest='max_original_sent_len', action="store",default=cfg.max_original_sent_len,
    #                     help="")
    parser.add_argument("--max_length", dest='max_length', action="store",default=12,
                        help="")

    ###################################################
    # configuration of decoder
    parser.add_argument("--batch_size", dest='batch_size', action="store",default=128,type=int,
                        help="")
    parser.add_argument("--epochs", dest='epochs', action="store",default=1000,type=int,
                        help="")
    parser.add_argument("--clip", dest='clip', action="store",default=1,type=int,
                        help="")
    parser.add_argument("--patience", dest='patience', action="store",default=50,type=int,
                        help="")
    parser.add_argument("--n_layers", dest='n_layers', action="store", default=1,type=int,
                        help="")
    parser.add_argument("--dropout", dest='dropout', action="store", default=0.4,type=int,
                        help="")
    parser.add_argument("--teacher_forcing_ratio", dest='teacher_forcing_ratio', action="store", default=0.75,type=float,
                        help="")

    parser.add_argument("--decoder_loss", dest='decoder_loss', action="store",default="sl",
                        help="")
    parser.add_argument("--decoder_net", dest='decoder_net', action="store",default="basicrnn",
                        help="")


    parser.add_argument("--decoder_path", dest='decoder_path', action="store",default=None,
                        help="")

    ###################################################
    # sentence embedding
    # parser.add_argument("--embed_method", dest='embed_method', action="store",default=cfg.embed_method,
    #                     help="")
    # parser.add_argument("--embed_mode", dest='embed_mode', action="store",default=cfg.embed_mode,
    #                     help="")
    parser.add_argument("--encoder_name", dest='encoder_name', action="store",default=None,
                        help="")
    parser.add_argument("--encoder_path", dest='encoder_path', action="store",default=None,
                        help="")


    ###################################################
    # save
    parser.add_argument("--result_dir", dest='result_dir', action="store",default=None,
                        help="")
    parser.add_argument("--model_save_path", dest='model_save_path', action="store",default=None,
                        help="path to decoder model")
    parser.add_argument("--sentence_save_path", dest='sentence_save_path', action="store",default=None,
                        help="path to generated sentences")


    return parser.parse_args()



################################################################################
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()



    ###################################################
    # TRAINING OR TESTING THE DECODING NETWORK
    run = opt.run
    print("==={}ing THE DECODING NETWORK===\n".format(run))

    if run == "train":

        configs = vars(opt)

        if not os.path.exists(opt.model_save_path):
            os.makedirs(opt.model_save_path)

        ###################################################
        # LOADING PRE-TRAINED EMBEDDING MODEL
        print("===LOADING PRE-TRAINED EMBEDDING MODEL===\n")
        # embed = embedding(method=opt.embed_method, mode=opt.embed_mode, model_name_or_path=opt.embed_model_name_or_path, device=device)
        encoder = encoding(model_name=opt.encoder_name, model_path=opt.encoder_path, device=device)

        ###################################################
        # LOADING DATASETS AND CREATING VOCABULARY FOR DECODER
        print("===LOADING DATASETS AND CREATING VOCABULARY FOR DECODER===\n")
        dataset = TextDataLoader(data_dir=opt.data_dir,
                                 train_filename=opt.train_filename,
                                 valid_filename=opt.valid_filename,
                                 test_filename=opt.test_filename,
                                 init_token=opt.init_token,
                                 eos_token=opt.eos_token,
                                 pad_token=opt.pad_token,
                                 batch_size=opt.batch_size,
                                 max_length=opt.max_length,
                                 device=device)

        vocab = dataset.generate_vocabulary()

        # save the vocabulary
        from vec2seq.utils import save_vocab

        vocab_save_path = opt.model_save_path + "/vocab.txt"
        save_vocab(vocab.vocab, vocab_save_path)

        train_set, valid_set, test_set = dataset.split()

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

        net = ConRNN if opt.decoder_net == "conrnn" else basicRNN

        decoder = net(vocab_size=OUTPUT_DIM, vector_dim=DEC_EMB_DIM, hid_dim=HID_DIM, n_layers=opt.n_layers,
                             loss=opt.decoder_loss,dropout=opt.dropout)
        decoder = decoder.to(device)
        print(decoder)
        print('The model has {} trainable parameters'.format(count_parameters(decoder)))

        #
        # if path.exists(opt.model_save_path): #+".pt"
        #     print("Error: duplicate name of the path for saving model.")
        #     sys.exit(0)

        decoder.apply(init_weights)

        valid_path = opt.data_dir + opt.valid_filename


        # save configurations
        configs["vocab_size"]=OUTPUT_DIM
        configs["vector_dim"]=DEC_EMB_DIM
        configs["hid_dim"]=HID_DIM


        with open(opt.model_save_path + "/config.json","w") as f:
            f.write(json.dumps(configs,ensure_ascii=False))


        trainer = trainer(decoder=decoder,encoder=encoder,train_set=train_set,valid_set=valid_set,valid_path=valid_path,batch_size=opt.batch_size,max_length=opt.max_length,
            corpora_field=vocab.vocab,loss_mode=opt.decoder_loss,
            epochs=opt.epochs,clip=opt.clip,patience=opt.patience,model_save_path=opt.model_save_path,
            device=device,teacher_forcing_ratio=opt.teacher_forcing_ratio,
                          init_token=opt.init_token,eos_token=opt.eos_token,pad_token=opt.pad_token)

        trainer.train()






    elif run == "test":
        # if path.exists(opt.save_path):
        #     print("Error: duplicate name of the path for saving results.")
        #     sys.exit(0)
        # device='cpu'
        cfgs = read_config(opt.decoder_path + '/config.json')


        encoder = encoding(model_name=cfgs["encoder_name"], model_path=cfgs["encoder_path"], device=device)


        from vec2seq.utils import read_vocab

        vocab = read_vocab(opt.decoder_path + '/vocab.txt')


        net = ConRNN if cfgs["decoder_net"] == "conrnn" else basicRNN
        # print(cfgs["decoder_net"])
        decoder=net(vocab_size=cfgs["vocab_size"], vector_dim=cfgs["vector_dim"], hid_dim=cfgs["hid_dim"],n_layers=cfgs["n_layers"], loss=cfgs["decoder_loss"])
        print(decoder)
        decoder.load_state_dict(torch.load(opt.decoder_path + '/checkpoint.pt',map_location='cuda:0'))
        decoder = decoder.to(device)


        test_path = opt.data_dir + opt.test_filename #

        test_set=[]
        with open(test_path) as f:
            for line in f:
                test_set.append(line.strip())

        print("Decoding...")
        results= test(encoder=encoder,decoder=decoder,decoder_cfg=cfgs,
            test_set=test_set,results_save_path=opt.sentence_save_path,
           corpora_field=vocab,device=device)

        scores = eval(results)

        print(tabulate(results, headers=["Reference","Generation"]))

        print(tabulate(scores.items()))


