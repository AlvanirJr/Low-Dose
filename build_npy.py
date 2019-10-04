# title                 :build_npy.py
# description           :It converts label images into .npy format
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


import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

"""
Parameters for converting output images into .npy files


low_quality_dir:    directory of the low quality images
high_quality_dir:   directory of the high quality images (the files must have the same names that those in the directory of low quality images)
target_dir:         directory where the .npy files will be saved
files_ext:          extension of images files at low_quality_dir and high_quality_dir
debug:              flag to allow intermediate visualization

"""


segmented_dir = '/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-GT/'
target_dir = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/15_projections-target/"
files_ext = '.png'

debug = False

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for file in os.listdir(segmented_dir):

        if file.endswith(files_ext):

            output_img = io.imread(os.path.join(segmented_dir,file), pilmode='F')

            #esse .npy aqui tem que ter 0 e 1 para cada uma das classes. Necessário checar ser esta assim.

            target = np.zeros(output_img.shape)
            #print(target.shape)
            temp = (output_img-np.min(output_img))/(np.max(output_img)-np.min(output_img)) # normaliza entre 0 e 1
            target[temp >= 0.5] = 1
            target[temp < 0.5] = 0

            if debug:
                print('min: {} max: {}'.format(np.min(target), np.max(target)))
                plt.figure()
                plt.imshow(target, cmap='gray', vmin=np.min(target), vmax=np.max(target))
                plt.show()
                break
            else:

                np.save(target_dir+file[0:len(file)-3]+'npy', target)
                print(target_dir+file+ " Done!")



