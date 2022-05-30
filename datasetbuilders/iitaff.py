import json
import os
import torch
import numpy as np
import pickle

from PIL import Image

def to_cat(y, num_classes=None, dtype=float):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


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

        for val_file in val_filenames:
            if val_file in train_filenames:
                train_filenames.remove(val_file)

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
        image = np.array(Image.open(image_path).convert("RGB"))

        # depth_path = f'{self.root}/deep/{filename[:-4]}.txt'
        # with open(depth_path) as file:
        #     raw_depth = file.read().split('\n')
        # raw_depth.remove("")
        # depth = np.array([np.array(row.split(' '), dtype=float) for row in raw_depth])

        mask_path = f'{self.root}/affordances_labels/{filename[:-4]}.txt'
        with open(mask_path) as file:
            raw_mask = file.read().split('\n')
        raw_mask.remove("")
        mask = np.array([np.array(row.split(' '), dtype=float) for row in raw_mask])

        object_path = f'{self.root}/object_labels/{filename[:-4]}.txt'
        with open(object_path) as file:
            raw_object = file.read().split('\n')
        raw_object.remove("")
        object = np.array([np.array(row.split(' ')[0], dtype=float) for row in raw_object])



        sample = dict(image=image, mask=mask, object=object) #depth=depth,


        image = np.array(Image.fromarray(sample["image"]).resize((244, 244), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((244, 244), Image.NEAREST))
        # depth = np.array(Image.fromarray(sample["depth"]).resize((244, 244), Image.NEAREST))



        mask = to_cat(mask, 10)
        object = to_cat(object, 10)
        if object.shape[0] > 1:
            object = [sum(object)]
        # object encodes the number of times each affordance is represented in each image. relationship aware module learns this

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] =  np.moveaxis(mask, -1, 0)
        # sample["depth"] = np.expand_dims(depth, 0)
        sample["object"] =  np.moveaxis(object, -1, 0)

        return sample
