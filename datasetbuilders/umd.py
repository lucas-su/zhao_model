import json
import os
import torch
import numpy as np
import pickle
from scipy.io import loadmat
import re
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

class umd(torch.utils.data.Dataset):
    def __init__(self, root,mode):

        self.root = root
        self.images_directories = []
        train_filenames = []
        test_filenames = []

        with open(f'{root}/category_split.txt', 'r') as file:
            splits = file.read().split('\n')
        splits.remove("")

        walk = list(os.walk(f"{self.root}/tools/"))
        names_per_object = []
        for folder in walk:
            names_per_object.extend([f"{folder[0].split('/')[-1]}/{file}" for file in folder[-1]])

        for row in splits:
            row = row.split(' ')
            r = re.compile(f".*^{row[1]}.*rgb\.jpg")
            if int(row[0]) == 1:
                train_filenames.extend([file[:-7] for file in list(filter(r.match, names_per_object))])
            else:
                test_filenames.extend([file[:-7] for file in list(filter(r.match, names_per_object))])

        if mode == 'train':
            self.filenames = train_filenames
        elif mode == 'test':
            self.filenames = test_filenames
        # elif mode == 'valid':
        #     self.filenames = val_filenames
        else:
            raise ValueError('file loading error: define mode failed')


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]



        image_path = f'{self.root}/tools/{filename}rgb.jpg'
        image = np.array(Image.open(image_path).convert("RGB"))

        # depth_path = f'{self.root}/tools/all/{filename[:-4]}.txt'
        # with open(depth_path) as file:
        #     raw_depth = file.read().split('\n')
        # raw_depth.remove("")
        # depth = np.array([np.array(row.split(' '), dtype=float) for row in raw_depth])

        mask_path = f'{self.root}/tools/{filename}label.mat'

        f = loadmat(mask_path)
        mask = f['gt_label']

        # with open(mask_path) as file:
        #     raw_mask = file.read().split('\n')
        # raw_mask.remove("")
        # mask = np.array([np.array(row.split(' '), dtype=float) for row in raw_mask])

        all_objects = ["knife", "saw", "scissors", "shears", "scoop", "spoon", "trowel", "bowl", "cup", "ladle",
                       "mug", "pot", "shovel", "turner", "hammer", "mallet", "tenderizer"]

        current_obj = ""

        for ob in all_objects:
            if ob in filename:
                current_obj = all_objects.index(ob)

        assert current_obj != ""


        sample = dict(image=image, mask=mask, object=current_obj)

        image = np.array(Image.fromarray(sample["image"]).resize((244, 244), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((244, 244), Image.NEAREST))
        # depth = np.array(Image.fromarray(sample["depth"]).resize((244, 244), Image.NEAREST))

        mask = to_cat(mask, 7)
        object = to_cat(current_obj, 17)
        # if object.shape[0] > 1:
        #     object = [sum(object)]
        # object encodes the number of times each affordance is represented in each image. relationship aware module learns this

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] =  np.moveaxis(mask, -1, 0)
        # sample["depth"] = np.expand_dims(depth, 0)
        sample["object"] =  np.moveaxis(object, -1, 0)

        return sample
