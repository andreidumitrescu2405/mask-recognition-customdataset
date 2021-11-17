"""
DatasetPath -> {train: [imag1, imag2...], test: [imag50, imag51,...], validation: [imag100, imag101,...]}

"""

import os
import random
import json

if __name__ == "__main__":
    # Input data
    dataset_path = 'Data/faces_dataset'
    split_ration = [0.7, 0.15, 0.15]

    # Dict for storing the splits
    split = {"train": [], "test": [], "val": []} 

    # Iterate throught folders
    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)
        print(class_folder_path)

        # All images from current class
        all_images = os.listdir(class_folder_path)

        train_idx = int(split_ration[0] * len(os.listdir(class_folder_path)))
        val_idx = int(split_ration[1] * len(os.listdir(class_folder_path)))
        test_idx = int(split_ration[2] * len(os.listdir(class_folder_path)))

        # Shuffle the list
        random.shuffle(all_images)
        
        # Remove extensions
        all_images = [x.replace(".png", "") for x in all_images]

        # Split the list
        train_temp = all_images[0:train_idx]
        val_temp = all_images[train_idx: train_idx + val_idx]
        test_temp = all_images[train_idx + val_idx:]

        # Append to main dict/json
        split['train'].extend(train_temp)
        split['val'].extend(val_temp)
        split['test'].extend(test_temp)

    # Save json with splits
    with open('split.json', 'w') as f:
        json.dump(split, f)








