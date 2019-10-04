import os
from skimage import data, img_as_float
from skimage.measure._structural_similarity import compare_ssim as ssim
import skimage.io as io

projs = 15
experiment_dir_simple_seg = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256 LOW-DOSE/{}_projections-GT/".format(projs)
GT_dir = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256-GT/"

experiment_dir_cnn = "/home/andrei/Área de Trabalho/Pesquisa/public-segmentation-CNN/results-VGG-UNET-{}-projs/".format(projs)
GT_dir_cnn = "/home/andrei/Área de Trabalho/Pesquisa/public-segmentation-CNN/results-VGG-UNET-{}-projs/".format(projs)

tomo_ssim = []
general_ssim = []
general_cnn_ssim = []
general_ss_ssim = []
def SSIM_neural_output():
    media = 0
    for  root, folders, _ in os.walk(experiment_dir_cnn):
        cont = 0

        for z in folders:
            im_experiment = io.imread(os.path.join(experiment_dir_cnn, z +"/final_rec.png"))
            im_GT = io.imread(os.path.join(GT_dir_cnn, z + "/target.png"))
            ssim_error = ssim(im_experiment, im_GT)
            general_ssim.append(ssim_error)
            # print("Done {}".format(100 * cont / len(folders)))
            cont = cont + 1
            # print("general ssim {}".format(ssim_error))
            # print("{} of {} folders visited".format(cont, len(folders)))

        media = (sum(general_ssim)/cont)
        break
    return media


def SSIM():
    media = 0
    for  root, _, files in os.walk(experiment_dir_simple_seg):
        cont = 0

        for z in files:
            # print(z)
            im_experiment = io.imread(os.path.join(experiment_dir_simple_seg, z))
            im_GT = io.imread(os.path.join(GT_dir, z))
            ssim_error = ssim(im_experiment, im_GT)
            general_ssim.append(ssim_error)
            # print("Done {}".format(100 * cont / len(files)))
            cont = cont + 1
            # print("general ssim {}".format(ssim_error))
        media = (sum(general_ssim)/cont)
    return media

def SSIM_between_classifiers():
    for  root, folders, _ in os.walk(experiment_dir_cnn):
        cont = 0

        for z in folders:
            im_experiment_cnn = io.imread(os.path.join(experiment_dir_cnn, z +"/final_rec.png"))
            im_GT = io.imread(os.path.join(GT_dir_cnn, z + "/target.png"))
            ssim_error = ssim(im_experiment_cnn, im_GT)
            general_cnn_ssim.append(ssim_error)

            im_experiment_ss = io.imread(os.path.join(experiment_dir_simple_seg, z +"png"))
            ssim_error = ssim(im_experiment_ss, im_GT)
            general_ss_ssim.append(ssim_error)

            print("Done {}".format(100 * cont / len(folders)))
            cont = cont + 1
            # print("general ssim {}".format(ssim_error))
            # print("{} of {} folders visited".format(cont, len(folders)))

        media_cnn = (sum(general_cnn_ssim)/cont)
        media_ss = (sum(general_ss_ssim)/cont)
        break
    return {'cnn': media_cnn, 'ss': media_ss}

if __name__ == '__main__':
    seg = SSIM_between_classifiers()
    simple_seg = seg['ss']
    cnn_seg = seg['cnn']
    print("Simple Segmentation SSIM result: {}%".format(simple_seg)) # 0.8008813164296595
    print("1 axis CNN Segnmentation SSIM result: {}%".format(cnn_seg)) # 0.9384656959066952