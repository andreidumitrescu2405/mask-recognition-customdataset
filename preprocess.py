import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
0. Transformare in grayscale
1. Reducerea rezolutiei la [rezX, rezY]
2. Normalizarea 0-1
"""

if __name__ == "__main__":
    # Inputs
    dataset_path = "./Data/faces_dataset"
    output_path = "./Data/faces_dataset_preprocessed"
    desired_rezolution = (64, 64)

    # Make output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Iterate all classes
    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)

        # Iterate all images in each folder
        for file in os.listdir(class_folder_path):
            file_full_path = os.path.join(class_folder_path, file)

            # Read the image and convert to grayscale
            img = cv2.imread(file_full_path, 0)

            # Resize
            img_resized = cv2.resize(img, desired_rezolution)

            # Normalizare
            norm_img = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))

            # Save data
            output_path_save = os.path.join(output_path, file.replace(".png", ".npz"))
            
            # 0, 1, 2
            gr_int = 0 if class_folder=="nomask" else (1 if class_folder=="maskwrong" else 2)

            # if class_folder=="nomask":
            #     gr_int = 0
            # elif class_folder=="maskwrong":
            #     gr_int = 1
            # else:
            #     gr_int = 2

            np.savez(output_path_save, img=norm_img, gr=gr_int)

            print(f"Saved {output_path_save}")
