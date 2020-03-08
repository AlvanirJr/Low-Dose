import matplotlib.pyplot as plt
import os
import skimage.io as io
import skimage
import scipy.misc as misc
projs = 15
# src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/{}_projections/".format(projs)
# dest = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/{}_projections-GT/".format(projs)
src = "/home/andrei/low-dose/DATASET-REGULARIZED/"
dest = "/home/andrei/low-dose/DATASET-GT/"
cont = 0
for  root, _, files in os.walk(src):

    for z in files:

        im = io.imread(os.path.join(src,z))
        seg = im > 175
        io.imsave(os.path.join(dest,z), skimage.img_as_ubyte(seg*255))

        print("Done {}".format(100*cont/len(files)))
        cont = cont + 1







