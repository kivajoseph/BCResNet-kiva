#  Utilities
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse
import math



import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn

import random

## General pytorch libraries
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.autograd import Variable
from torchinfo import summary

## Import audio related packages
# import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import soundfile as sf
import soundfile as s
from helper_functions import set_seed, count_model_parameters
from helper_functions import naive_lr_decay

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

############ Get functions from BCResnet helper implementation ########
from model.py import (
    BCResNet,
    BroadcastedBlock,
    SubSpectralNorm,
    TransitionBlock
)

##################################################################################################
# Add arguments you want to pass via commandline
##################################################################################################
parser = argparse.ArgumentParser(description='Marteethi :: Classifier design')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--init_lr', default=0.001, type=float,
                    metavar='N',
                    help='Initial learning rate with default set based on Adam optimiser\n.'
                    'If you change the lr decay schedule, then update the init_lr accordingly')
parser.add_argument('--num_epochs', default=120, type=int,
                    metavar='N',
                    help='Total number of training epochs')
parser.add_argument('--model_comments', default='model training dummy comment', type=str,
                    metavar='N',
                    help='TB: Comment')
parser.add_argument('--model_chk_pth', default='/home/mohapatrapayal/marteethi/model_chkpts/dummy/', type=str,
                    metavar='N',
                    help='Give the path to checkpoint file')
parser.add_argument('--cuda_pick', default='cuda:3', type=str,
                    metavar='N',
                    help='Give the cuda device id')
# parser.add_argument('--model_type', default='temp_compr', type=str,
#                     metavar='N',
#                     help='input which model you want to test \n'
#                           'temp_compr = Uses the 1D CNN based model, ~10M parameters \n'
#                           )
args = parser.parse_args()

cuda_pick = args.cuda_pick
num_epochs = args.num_epochs
init_lr = args.init_lr
batch_size = args.batch_size
model_comments = args.model_comments
model_chk_pth = args.model_chk_pth
## Create a directory if its not existing
if not os.path.exists(model_chk_pth):
    os.makedirs(model_chk_pth)

##################################################################################################
# GPU related + Tensorboard writer
##################################################################################################
set_seed(2711)
device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
print("Device used is ",device)
writer = SummaryWriter(comment=model_comments)

##################################################################################################
# DATA LOADER ##
# split data and translate to dataloader
##################################################################################################

def load_speechcommands(root='./', download=True):
    # Assuming the root directory is where the datasets are saved/downloaded
    full_dataset = SPEECHCOMMANDS(root=root, download=download, subset='training')

    # Shuffle and split the dataset into training and validation sets
    total_size = len(full_dataset)
    indices = list(range(total_size))
    split = int(np.floor(0.1 * total_size))  # For example, 10% for validation
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    return train_dataset, val_dataset

# Load the datasets
train_dataset, val_dataset = load_speechcommands(root='/datasets/speech_commands/')

# For the test set, you can directly use the subset parameter
test_dataset = SPEECHCOMMANDS(root='/datasets/speech_commands/', download=True, subset='testing')


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Typically, you don't need to shuffle the validation set
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



##################################################################################################
## BCResNet Model
## Paper :: https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Cao_59_t1.pdf
## Diff :: D47117321
##################################################################################################


##################################################################################################
## Define the model and training helper functions
##################################################################################################
model = BCResNet(device=device, num_class = 3 , c=4, FINnorm=True, lastAct=None).to(device).to(torch.double)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)

train_steps = len(train_loader)
valid_steps = len(valid_loader)

print('Number of trainable model parameters are --> ', count_model_parameters(model))

def train(epoch, train_loader):
    avg_loss = 0.0
    model.train()
    for i, (signals, label) in enumerate(train_loader):
        signals = signals.to(device)
        label = label.to(device).to(torch.long)
        model_output = model(signals)
        loss = criterion(model_output, label)
        avg_loss = (avg_loss * i + loss.item())/(i+1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}/{train_steps}], avg_Loss: {avg_loss:.4f}. \n')
    writer.add_scalar("Model loss/train", avg_loss, epoch)
    writer.flush()

def evaluate(epoch, devel_loader):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for i, (signals, label) in enumerate(devel_loader):

            signals = signals.to(device)
            label = label.to(device).to(torch.long)
            model_output = model(signals)
            loss = criterion(model_output, label)
            avg_loss = (avg_loss * i + loss.item())/(i+1)
            y_valid = torch.argmax(model_output, axis=1)
            bal_acc = balanced_accuracy_score(y_valid.cpu(), label.cpu())
            f1_score_cal = f1_score(label.cpu(), y_valid.cpu(), average=None)
            cm  = confusion_matrix(label.cpu(), y_valid.cpu())
            if (i+1) % 1 == 0:
                print(f'Eval Epoch [{epoch+1}], Step [{i+1}/{valid_steps}], avg_Loss: {avg_loss:.4f}, bal_acc : {bal_acc:.4f}., f1 : {f1_score_cal}\n')
                print(f'cm = {cm}\n')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


        writer.add_scalar("Model loss/val", avg_loss, epoch)
        writer.add_scalar("Model balanced acc/val", bal_acc, epoch)
        writer.add_scalar("F1 Class 0/val", f1_score_cal[0], epoch)
        writer.add_scalar("F1 Class 1/val", f1_score_cal[1], epoch)
        writer.add_scalar("F1 Class 2/val", f1_score_cal[2], epoch)
        writer.flush()

##################################################################################################
## Launch training and evaluation
##################################################################################################

for epoch in range(num_epochs):
    naive_lr_decay(optimizer, init_lr, epoch, num_epochs)
    train(epoch, train_loader)
    # print(get_lr(optimizer))
    torch.save(model.state_dict(),model_chk_pth + f'epoch_{epoch+1}.pth')
    evaluate(epoch, valid_loader)

# Do an evaluation with test set
with torch.no_grad():
    model.eval()
    avg_loss = 0
    for i, (signals, label) in enumerate(test_loader):
        # print(i)
        signals = signals.to(device)
        label = label.to(device).to(torch.long)
        model_output = model(signals)
        y_test = torch.argmax(model_output, axis=1)
        test_acc = balanced_accuracy_score(y_test.cpu(), label.cpu())
        f1_score_cal = f1_score(label.cpu(), y_test.cpu(), average=None)
        cm  = confusion_matrix(label.cpu(), y_test.cpu())

print(y_test)
print(label)
print("Balanced Test accuracy is ", test_acc)
print("F1 score is ", f1_score_cal)
print("cm = ",cm)
