
from DataInterface import DataInterface
import random
from math import floor
import numpy as np

src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256/"
prop = [.8, .1, .1]

data = DataInterface(src)

scans = data.get_tomo_list()
random.shuffle(scans)

training = scans[0:floor(len(scans)*prop[0])]
validation = scans[floor(len(scans)*prop[0]):floor(len(scans)*prop[0])+floor(len(scans)*prop[1])]
test = scans[floor(len(scans)*prop[0])+floor(len(scans)*prop[1]):len(scans)]


np.save("training", training)
np.save("validation", validation)
np.save("test", test)