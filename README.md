# Low-Dose
Segmentação de imagens low-dose, aplicada com CNN



Execute the following:

#Procedure to handle the data

1. run segmentation_test.py to obtain the dataset segmented (ground truth)
2. run create-low-dose-data.py to obtain Low Dose CBCT reconstructions of dataset
3. run training-validation-test.py to create disjoint groups of Tomos for training, validate, and executing the Neural Net




#Procedure to segment using the neural net

1. run build_npy.py to obtain the correct format of output files
2. run data_utils.py to create csv files for training, validation, and test
3. adjust the parameters in data_loader.py
4. run train.py to train the model
5. run generalization.py to save the output of the images in the test set
