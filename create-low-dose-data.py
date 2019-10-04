import os
from DataInterface import DataInterface
from ConeBeamCT import ConeBeamCT
import skimage.io as io
import sklearn.preprocessing as normalize
import skimage
import numpy
import scipy.misc as misc

nr_projections = 15

src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256/"
dest = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/"



new_dest = os.path.join(dest,"{}_projections/".format(nr_projections))
try:

    os.mkdir(new_dest)

    data = DataInterface(src)
    scans = data.get_tomo_list()

    t = 0
    for x in scans:
        vol = data.get_tomo_volume(x)
        ct = ConeBeamCT(vol)
        rec = ct.run_new_scan(nr_projections)
        maxim = rec.max()
        _, _, z = rec.shape

        for slice in range(z):
            im = rec[:,:,slice]
            # print(im[150])
            # im = (normalize.normalize(im))
            # im *= 255 / maxim
            # print(skimage.img_as_ubyte(im)* m)
            # io.imsave(os.path.join(new_dest, "Tomo_{}_slice_{}.png".format(str(x).zfill(3),str(slice).zfill(3))), skimage.img_as_ubyte(im)* maxim)
            misc.imsave(os.path.join(new_dest, "Tomo_{}_slice_{}.png".format(str(x).zfill(3),str(slice).zfill(3))), (im))
        print(100*t/len(scans))
        t = t + 1

except FileExistsError:
    print("PASTA JA EXISTENTE, ARQUIVO PROCESSADO")