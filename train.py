# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader import Tomographic_Dataset

from UNET import VGGNet, UNETmodel, UNETsota

import numpy as np
import time

import os

view = 1
projs = 15
net = "VGG-UNET"

batch_size = 40 #antes 10
epochs     = 100

pre = False
chkepoch = 0
momentum   = 0.5
w_decay    = 0 #antes 1e-5

#after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr         = 0.0001 # antes le-4
step_size  = 10
gamma      = 0.5
filename = "checkpoint.pth.tar"



configs         = "{}-model-{}-projs".format(net,projs)
n_class         = 2
train_file      = "training.csv"
val_file        = "validation.csv"
input_dir       = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections/".format(projs)
target_dir      = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections-target/".format(projs)


# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, configs)


if view == 1:
    print("Axial")
    input_dir = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections/".format(projs)
    target_dir = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections-target/".format(projs)
elif view == 2:
    print("COronal")
    input_dir = "/home/andrei/low-dose/DATASET-256-CORONAL LOW-DOSE/{}_projections/".format(projs)
    target_dir = "/home/andrei/low-dose/DATASET-256-CORONAL LOW-DOSE/{}_projections-target/".format(projs)
else:
    print("Sagittal")
    input_dir = "/home/andrei/low-dose/DATASET-256-SAGITTAL LOW-DOSE/{}_projections/".format(projs)
    target_dir = "/home/andrei/low-dose/DATASET-256-SAGITTAL LOW-DOSE/{}_projections-target/".format(projs)

validation_accuracy = np.zeros((epochs,1))

model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

if view == 1:
    print("Axial")
    model_src = "./models/{}-model-{}-projs".format(net, projs)
elif view == 2:
    print("Coronal")
    model_path = model_path + "Coronal"
    model_src = "./models/{}-model-{}-projsCoronal".format(net, projs)
else:
    print("Sagittal")
    model_path = model_path + "Sagittal"
    model_src = "./models/{}-model-{}-projsSagittal".format(net, projs)


print("GPU Available: ",use_gpu, " number: ",len(num_gpu))

train_data = Tomographic_Dataset(csv_file=train_file, phase='train', train_csv=train_file, input_dir=input_dir, target_dir=target_dir)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

#directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
val_data = Tomographic_Dataset(csv_file=val_file, phase='val', flip_rate=0, train_csv=train_file, input_dir=input_dir, target_dir=target_dir)
val_loader = DataLoader(val_data, batch_size=1, num_workers=4)

vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
fcn_model = UNETmodel(pretrained_net=vgg_model)
#fcn_model = UNETsota()


if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)



def train(chkepoch):
    if pre:
        openChekpoint(epochs, chkepoch)
    for epoch in range(epochs):
        scheduler.step()

        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(chkepoch, iter, loss.item()))
                #print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))
        
        print("Finish epoch {}, time elapsed {}".format(chkepoch, time.time() - ts))
        state = {'epoch': epoch + 1, 'state_dict': fcn_model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'losslogger': loss.item()}
        chkepoch +=1
        torch.save({
            'epoch': chkepoch,
            'model_state_dict': fcn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict()}, filename)
        torch.save(fcn_model, model_path)
        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_mse = []

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_mse.append(mse_acc(p, t))


    mse_accs = np.mean(total_mse)
    validation_accuracy[epoch] = mse_accs

    print("epoch{}, mse_acc: {}".format(epoch,mse_accs))


def mse_acc(pred, target):

    return np.mean(np.square(pred-target))

def openChekpoint(epochs, chk):
    checkpoint = torch.load(filename)
    fcn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epochs = epochs - checkpoint['epoch']
    chk = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Checkpoint loss:" + str(loss))
    print("Epochs left:" + str(epochs))


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    start = time.time()
    train(chkepoch)
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))

    np.save('validation_accuracy_{}-model-{}-projs.npy'.format(net,projs), validation_accuracy)
