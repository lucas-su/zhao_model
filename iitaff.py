import json
import os
import torch
import numpy as np
import pickle

from PIL import Image

class iitaff(torch.utils.data.Dataset):
    def __init__(self, root,mode):

        self.root = root
        self.images_directories = []

        with open(f'{root}/test.txt', 'r') as file:
            test_filenames = file.read().split('\n')
        with open(f'{root}/train_and_val.txt', 'r') as file:
            train_filenames = file.read().split('\n')
        with open(f'{root}/val.txt', 'r') as file:
            val_filenames = file.read().split('\n')

        train_filenames.remove("")
        test_filenames.remove("")
        val_filenames.remove("")

        if mode == 'train':
            self.filenames = train_filenames
        elif mode == 'test':
            self.filenames = test_filenames
        elif mode == 'valid':
            self.filenames = val_filenames
        else:
            raise ValueError('file loading error: define mode failed')


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = f'{self.root}/rgb/{filename}'
        depth_path = f'{self.root}/deep/{filename[:-4]}.txt'
        mask_path = f'{self.root}/affordances_labels/{filename[:-4]}.txt'

        image = np.array(Image.open(image_path).convert("RGB"))
        with open(depth_path) as file:
            raw_depth = file.read().split('\n')
        raw_depth.remove("")
        depth = np.array([np.array(row.split(' '), dtype=float) for row in raw_depth])
        with open(mask_path) as file:
            raw_mask = file.read().split('\n')
        raw_mask.remove("")
        mask = np.array([np.array(row.split(' '), dtype=float) for row in raw_mask])


        sample = dict(image=image, mask=mask, depth=depth)


        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        depth = np.array(Image.fromarray(sample["depth"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["depth"] = np.expand_dims(depth, 0)

        return sample
