import re, os
import numpy as np
import skimage.io as io
import pylab
import random
from math import floor
import scipy.misc as misc


class DataInterface():

    def __init__(self, data):
        self.__data_src = data

    def get_tomo_list(self):
        all = []
        for root, _, files in os.walk(self.__data_src):
            for z in files:
                tomo, _ = re.findall('\d+', z)
                all.append(int(tomo))

        return np.unique(all)

    def get_tomo_volume(self, index):
        slices = []
        for root, _, files in os.walk(self.__data_src):
            for z in files:
                tomo, _ = re.findall('\d+', z)

                if int(tomo) == index:
                    slices.append(z)

        slices.sort()
        z = len(slices)
        x, y = io.imread(os.path.join(self.__data_src,slices[0])).shape
        volume = np.zeros([x,y,z])
        k = 0

        for s in slices:
            im = io.imread(os.path.join(self.__data_src,s))
            volume[:,:,k] = im/256.0
            k = k + 1
        return volume

    def get_train_val_test(self, prop):

        if os.path.isfile("training_all.npy") and os.path.isfile("validation_for_net.npy") and os.path.isfile("test.npy"):
            print("Loading train-val-test division already sorted before!")
            training = np.load("training_all.npy")
            validation = np.load("validation_for_net.npy")
            test = np.load("test.npy")
        else:
            scans = self.get_tomo_list()
            random.shuffle(scans)

            training = scans[0:floor(len(scans) * prop[0])]
            validation = scans[floor(len(scans) * prop[0]):floor(len(scans) * prop[0]) + floor(len(scans) * prop[1])]
            test = scans[floor(len(scans) * prop[0]) + floor(len(scans) * prop[1]):len(scans)]

            np.save("training", training)
            np.save("validation", validation)
            np.save("test", test)

        training_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in training]
        validation_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in validation]
        testing_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in test]

        return (training_files, validation_files, testing_files)

    def get_axis_coronal(self):
        global src_segmented
        global src_segmented_coronal

        tomografias = []
        for root, _, files in os.walk(src_segmented):
            for z in files:
                linha = z.split("_")
                tomografias.append(int(linha[1]))
        tomografias = set(tomografias)
        for tomo in tomografias:
            volume = self.get_tomo_volume(tomo)
            for j in range(len(volume)):
                im = np.array(volume[:, j, :])
                m, n = im.shape
                out = im
                if(m % 32 != 0):
                    out = np.zeros((m+ (32 - (m % 32)), n ), dtype=im.dtype)
                    out[:m, :] = im
                if(n % 32 != 0):
                    out = np.zeros((m, n + (32 - (n % 32))), dtype=im.dtype)
                    out[:, :n] = im
                misc.imsave(os.path.join(src_segmented_coronal, "Tomo_{}_slice_{}.png".format(str(tomo).zfill(3),str(j).zfill(3))), out)


    def get_axis_sagittal(self):
        global src_lowdose
        global src_lowdose_sagittal

        tomografias = []
        for root, _, files in os.walk(src_lowdose):
            for z in files:
                linha = z.split("_")
                tomografias.append(int(linha[1]))
        tomografias = set(tomografias)
        for tomo in tomografias:
            volume = self.get_tomo_volume(tomo)
            for j in range(len(volume)):
                im = np.array(volume[j, :, :])
                m, n = im.shape
                out = im
                if (m % 32 != 0):
                    out = np.zeros((m + (32 - (m % 32)), n), dtype=im.dtype)
                    out[:m, :] = im
                if (n % 32 != 0):
                    out = np.zeros((m, n + (32 - (n % 32))), dtype=im.dtype)
                    out[:, :n] = im

                misc.imsave(os.path.join(src_lowdose_sagittal, "Tomo_{}_slice_{}.png".format(str(tomo).zfill(3), str(j).zfill(3))),
                          out)



if __name__ == "__main__":

    src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256/"
    src_coronal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-CORONAL/"
    src_sagittal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-SAGITTAL/"

    src_segmented = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-GT/"
    src_segmented_coronal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-CORONAL-GT/"
    src_segmented_sagittal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-SAGITTAL-GT/"

    src_lowdose = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/15_projections/"
    src_lowdose_coronal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-CORONAL LOW-DOSE/15_projections/"
    src_lowdose_sagittal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-SAGITTAL LOW-DOSE/15_projections/"

    src_lowdose_gt = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/15_projections/"
    src_lowdose_gt_coronal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-CORONAL LOW-DOSE/15_projections/"
    src_lowdose_gt_sagittal = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-SAGITTAL LOW-DOSE/15_projections/"

    dataset = DataInterface(src)
    dataset.get_axis_coronal()
    # dataset.get_axis_sagittal()
    #print(dataset.get_tomo_list())
    volume = dataset.get_tomo_volume(0)
    print(volume.shape)
    pylab.figure(2)
    pylab.imshow(volume[50, :, :])
    pylab.show()

