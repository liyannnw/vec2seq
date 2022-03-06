#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import torch.optim as optim
import time
from VectorDecoder.pytorchtools import EarlyStopping
from VectorDecoder.train import epoch_time
import math
from pathlib import Path
from argparse import ArgumentParser

from analogical_dataset_loader import AnalogicalDataLoader
from networks import LinearFCN
import torch.nn as nn


import solver_config as as_cfg
################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"


__usage__='''
python3 train_AnalogySolverinSpace.py --data_dir_path [DIR_PATH] --model_save_path [SAVE_PATH]
'''

################################################################################
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def dataFile_loader(data_dir,data="train",device="cpu"):


    data_path = [str(x) for x in Path(data_dir).glob("**/*.{}.npz".format(data))]
    # valid_path = [str(x) for x in Path(data_dir).glob("**/*.valid.npz")]

    if data == "test":
        test_mode = True
    else:
        test_mode = False

    data_set = AnalogicalDataLoader(data_path[0], test_mode=test_mode,device=device)
    # valid_set = AnalogicalDataLoader(valid_path[0], test_mode=False,device=device)
    # test_set = AnalogicalDataLoader(test_path, test_mode=True,device=device)

    return data_set#,valid_set


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
        torch.save(model.state_dict(), model_path)
        print("Saved")

################################################################################
def read_argv():
    parser=ArgumentParser(usage=__usage__)
    parser.add_argument("--data_dir_path",action="store",default=as_cfg.data_dir_path,
                        dest="data_dir_path",help="...")
    parser.add_argument("--model_save_path",action="store",default=as_cfg.model_save_path,
                        dest="model_save_path",help="...")


    parser.add_argument("--compose_mode",action="store",default=as_cfg.compose_mode,
                        dest="compose_mode",help="...")
    parser.add_argument("--nb_output_item",action="store",default=as_cfg.nb_output_item,type=int,
                        dest="nb_output_item",help="...")

    parser.add_argument("--layers_size",action="store",default=as_cfg.layers_size,
                        dest="layers_size",nargs='+', type=int,help="...")

    parser.add_argument("--batch_size",action="store",default=as_cfg.batch_size,type=int,
                        dest="batch_size",help="...")
    parser.add_argument("--patience",action="store",default=as_cfg.patience,type=int,
                        dest="patience",help="...")
    parser.add_argument("--epochs",action="store",default=as_cfg.epochs,type=int,
                        dest="epochs",help="...")


    return parser.parse_args()


################################################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = read_argv()

    print(">> Loading datasets...")
    data_dir_path = opt.data_dir_path
    train_set = dataFile_loader(data_dir_path,data="train",device=device)
    valid_set = dataFile_loader(data_dir_path, data="valid", device=device)

    print(">> Employing a network for solving sentence analogies in vector space...")
    compose_mode = opt.compose_mode#"concat"
    layers_size = opt.layers_size
    output_n = opt.nb_output_item
    model = LinearFCN(vector_size=train_set.vector_size(),layers_size=layers_size,output_n=output_n,compose_mode=compose_mode)
    print(model)

    model = model.to(device)
    model.apply(init_weights)
    criterion = nn.MSELoss()
    model_save_path=opt.model_save_path

    batch_size = opt.batch_size#128
    patience = opt.patience#50
    epochs = opt.epochs#1500

    print(">> Training...")
    train(train_set,valid_set,model,criterion,
          batch_size=batch_size,patience=patience,
          epochs=epochs,model_save_path=model_save_path)

