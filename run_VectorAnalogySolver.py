#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import torch.optim as optim
import time
from vec2seq.pytorchtools import EarlyStopping
from vec2seq.train import epoch_time
import math
from pathlib import Path
from argparse import ArgumentParser

from utils import AnalogicalDataLoader
from networks import LinearFCN
import torch.nn as nn

from vec2seq.networks import init_weights

import os
import json

################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"


__usage__='''

'''

################################################################################
def train(train_set,valid_set,model,criterion,batch_size=32,patience=50,epochs=100,model_save_path=None):



    train_generator = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_generator = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss=0
        for index,(batch,truth) in enumerate(train_generator):

            optimizer.zero_grad()

            output = model(batch)
            loss = criterion(output,truth)
            loss.backward()

            optimizer.step()
            train_loss +=loss.item()

        train_loss = train_loss/(index+1)

        model.eval()
        valid_loss=0

        with torch.no_grad():
            for index,(batch,truth) in enumerate(valid_generator):

                output = model(batch)
                loss = criterion(output, truth)

                valid_loss += loss.item()

            valid_loss = valid_loss/(index+1)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Epoch: {} | Time: {}m {}s'.format(epoch,epoch_mins,epoch_secs))
        print('\tTrain Loss: {:.5f} | Train PPL: {:7.3f}'.format(train_loss,math.exp(train_loss)))
        print('\t Val. Loss: {:.5f} |  Val. PPL: {:7.3f}'.format(valid_loss,math.exp(valid_loss)))

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if model_save_path:
        model_path = model_save_path#+".pt"
        torch.save(model.state_dict(), model_path+'/checkpoint.pt')
        print("Saved")

################################################################################
def read_argv():
    parser=ArgumentParser(usage=__usage__)


    parser.add_argument("--train_filepath",action="store",default=None,
                        dest="train_filepath",help="...")
    parser.add_argument("--valid_filepath",action="store",default=None,
                        dest="valid_filepath",help="...")

    parser.add_argument("--model_save_path",action="store",default=None,
                        dest="model_save_path",help="...")


    parser.add_argument("--compose_mode",action="store",default='ap',
                        dest="compose_mode",help="...")
    parser.add_argument("--output_n",action="store",default=1,type=int,
                        dest="output_n",help="...")

    parser.add_argument("--layers_size",action="store",default=[512,512,512],
                        dest="layers_size",help="...") #nargs='+', type=int,

    parser.add_argument("--batch_size",action="store",default=128,type=int,
                        dest="batch_size",help="...")
    parser.add_argument("--patience",action="store",default=50,type=int,
                        dest="patience",help="...")
    parser.add_argument("--epochs",action="store",default=1500,type=int,
                        dest="epochs",help="...")


    return parser.parse_args()


################################################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()

    configs = vars(opt)

    if not os.path.exists(opt.model_save_path):
        os.makedirs(opt.model_save_path)

    print(">> Loading datasets...")
    # data_dir_path = opt.data_dir_path

    train_set = AnalogicalDataLoader(opt.train_filepath, test_mode=False,device=device)
    valid_set = AnalogicalDataLoader(opt.valid_filepath, test_mode=False, device=device)

    configs["vector_size"]=train_set.vector_size()

    with open(opt.model_save_path + "/config.json", "w") as f:
        f.write(json.dumps(configs, ensure_ascii=False))
    #
    # train_set = dataFile_loader(data_dir_path,data="train",device=device)
    # valid_set = dataFile_loader(data_dir_path, data="valid", device=device)

    print(">> Employing a network for solving sentence analogies in vector space...")
    # compose_mode = opt.compose_mode#"concat"
    # layers_size = opt.layers_size
    # output_n = opt.output_n

    model = LinearFCN(vector_size=train_set.vector_size(),layers_size=opt.layers_size,output_n=opt.output_n,compose_mode=opt.compose_mode)
    print(model)

    model = model.to(device)
    model.apply(init_weights)
    criterion = nn.MSELoss()
    # model_save_path=opt.model_save_path
    #
    # batch_size = opt.batch_size#128
    # patience = opt.patience#50
    # epochs = opt.epochs#1500

    print(">> Training...")
    train(train_set,valid_set,model,criterion,
          batch_size=opt.batch_size,patience=opt.patience,
          epochs=opt.epochs,model_save_path=opt.model_save_path)


