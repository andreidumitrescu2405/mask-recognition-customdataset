import torch
import json
import os
import numpy as np

class BaselineDataloader(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, phase):
        self.phase = phase
        self.dataset_path = dataset_path
        self.split = split

        # Load dict
        with open(split, 'r') as f:
            split_dict = json.loads(f.read())

        self.files = split_dict[phase]
        print(f"Found {len(self.files)} in {phase}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load npz
        data = np.load(os.path.join(self.dataset_path, self.files[idx] + ".npz"))
        
        # Unpack
        img, gr = data['img'], data['gr']

        # Add channel dimension. Din (64, 64) -> (1, 64, 64)
        img = np.reshape(img, (1, img.shape[0], img.shape[1])).astype(np.float32)

        # From array to int
        gr = int(gr)
        
        return img, gr

    

