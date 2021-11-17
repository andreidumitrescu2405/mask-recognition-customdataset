# mask-recognition-customdataset
Mask Recognition Deep Learning Algorithm

This algorithm train itself on a custom data set to recognize medical masks. In my case my dataset is composed of 2400 images, 800 images of me wearing the mask correctly, 800 images of me wearing the mask incorrectly and 800 images of me not wearing the mask. As you can guess, the algorithm has 3 outputs: *mask*, *maskwrong* and *nomask*. I used the 70%-15%-15% data split. 70% of the images is used in training dataset, 15% in the validation dataset and the last 15% in the test dataset. You have to execute the code in the following order:
  -run split.py to split our data set into all 3 categories (training, validation, testing) and save them into the split.json file
  -run preprocess.py to preprocess our data where we convert our RGB images into one channel, gray scale, images, resize the photos to the desired rezolution (64 by 64 pixels), normalization where we obtain the intensity of the pixels from 0 to 255, saving the data and setting up the ground truth (0 - no mask, 1 - maskwrong, 2 - mask
  -run train.py where we save our experiments via comet_ml. (you have to set your own api_key and set up a project name and a workspace to see the progress of the training and validation). 
  -run test.py on the rest 15% of the images to see how well your Convolutional Neural Network has trained. As visualization we used f1_score and confusion_matrix as metrics from sklearn.metrics library
Make your own dataset of you wearing the mask correctly, incorrectly and not wearing it at all and try yourself.
