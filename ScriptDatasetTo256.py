import os
import skimage.io as io
import numpy as np
import scipy.misc as misc
src = "/home/andrei/low-dose/DATASET-256/"
src_256 = "/home/andrei/low-dose/DATASET-REGULARIZED/"
tomografias = []
for root, _, files in os.walk(src):
    for z in files:
        volume = io.imread(os.path.join(src,z))
        # print(int(z.split("_")[3][0:3]))
        # print(volume.shape)
        if int(z.split("_")[3][0:3]) > 256:
            continue
        if volume.shape != (256, 256):
            im = np.zeros((256,256))
            print(im)
            print(z)
        if int(z.split("_")[3][0:3]) < 256:
            z2 = int(z[15:18]) + 1
            z= z[0:15] + str(z2).zfill(3) + ".png"
            if not os.path.isfile(os.path.join(src, z)):
                for k in range(z2, 257):
                    im = np.zeros((256, 256))
                    z= z[0:15] + str(k).zfill(3) + ".png"
                    im = im.astype(np.uint8)
                    io.imsave(os.path.join(src_256,z), im)
        io.imsave(os.path.join(src_256,z), volume.astype(np.uint8))

