# title                 :data_utils.py
# description           :It creates a training-validation-test split and computes the mean gray value in the images of the training set
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-05-16
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# matplotlib_version    :3.0.3
# pilow_version         :6.0.0
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import re, os
import numpy as np
import pandas as pd
from math import floor
import scipy.misc
import random


"""
Parameters to split data into groups for training, validation, and testing  

src_img:            directory containing the set of low quality or high quality images 

"""


src_img = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/15_projections/"




def create_csv_files(src_data, list, csv_name):

    file = open(csv_name, 'w')
    list = np.load(list)

    for data in os.listdir(src_img):

        tomo, _ = re.findall('\d+', data)


        if int(tomo) in list:
            file.write(data + ',' + data[0:len(data) - 3] + 'npy' + '\n')

    file.close()

def data_mean_value(csv, dir):
    data = pd.read_csv(csv)
    r, c = data.shape
    values = np.zeros((r,3))
    idx = 0
    for i, row in data.iterrows():
        img = scipy.misc.imread(dir+row[0], mode="RGB")
        values[idx,:] = np.mean(img, axis=(0,1))
        idx += 1

    return np.mean(values,axis=0)



if __name__ == "__main__":
    create_csv_files(src_img, "training.npy", "training.csv")
    create_csv_files(src_img, "validation.npy", "validation.csv")
    create_csv_files(src_img, "test.npy", "test.csv")