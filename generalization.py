import torch
import time
import os
from torch.autograd import Variable
import skimage.io as io
from data_loader import Tomographic_Dataset
from torch.utils.data import Dataset, DataLoader
from data_utils import data_mean_value
import numpy as np
import ntpath
from matplotlib import pyplot as plt
from torchvision import utils
#from skimage.morphology import disk
#from skimage.filters.rank import median
import scipy.misc as misc


n_class         = 2
net             = 'VGG-UNET'
projs           = 15
input_dir       = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections/".format(projs)
target_dir      = "/home/andrei/low-dose/DATASET-256 LOW-DOSE/{}_projections-target/".format(projs)
means           = data_mean_value("training.csv", input_dir) / 255.

model_src = "./models/{}-model-{}-projs".format(net, projs)



def evaluate_img():

    test_data = Tomographic_Dataset(csv_file="validation.csv", phase='val', flip_rate=0, train_csv="test.csv",
                                    input_dir=input_dir, target_dir=target_dir)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

    fcn_model = torch.load(model_src)
    n_tests = len(test_data.data)

    print("{} files for testing....".format(n_tests))

    folder = "./results-{}-{}-projs/".format(net, projs)
    if not os.path.exists(folder):
        os.makedirs(folder)

    execution_time = np.zeros((n_tests, 1))
    count = 0
    for iter, batch in enumerate(test_loader):
        name = batch['file'][0]
        print(name)
        dest = os.path.join(folder, name[0:len(name)-3])
        if not os.path.exists(dest):
            os.mkdir(dest)

        input = Variable(batch['X'].cuda())
        start = time.time()
        output = fcn_model(input)
        end = time.time()
        elapsed = end-start
        execution_time[count] = elapsed
        print('execution: {} seconds'.format(elapsed))
        count = count + 1

        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        y = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)

        img_batch = batch['X']
        img_batch[:, 0, ...].add_(means[0])
        img_batch[:, 1, ...].add_(means[1])
        img_batch[:, 2, ...].add_(means[2])

        grid = utils.make_grid(img_batch)
        x = grid.numpy()[::-1].transpose((1, 2, 0))

        final_rec = y[0,:,:]

        original = io.imread(batch['o'][0], pilmode='RGB')

        io.imsave(dest+'/target.png', target[0,:,:])
        io.imsave(dest+'/final_rec.png', final_rec)
        io.imsave(dest+'/input.png', x)
        io.imsave(dest+'/original.png', original)



        print("executed {} of {}\n".format(iter,len(test_loader)))

    print("mean: {}".format(np.mean(execution_time[1:n_tests])))
    print("std: {}".format(np.std(execution_time[1:n_tests])))



if __name__ == "__main__":
    evaluate_img()

