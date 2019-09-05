import matplotlib.pyplot as plt
import os
from scipy import misc

projs = 15
src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/{}_projections/".format(projs)
dest = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/{}_projections-GT/".format(projs)
# src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256/"
# dest = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-GT/"
cont = 0

for  root, _, files in os.walk(src):

    for z in files:
        im = misc.imread(os.path.join(src,z))
        seg = im > 175
        misc.imsave(os.path.join(dest,z), seg*255)
        print("Done {}".format(100*cont/len(files)))
        cont = cont + 1








