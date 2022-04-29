import json
import os
import torch
import numpy as np
import pickle

from PIL import Image


if os.path.exists("devmode"):
    transfer_table = json.load(open('/media/luc/data/transfer_table.json','r'))
else:
    transfer_table = json.load(open('/home/schootuiterkampl/sunrgbd/transfer_table.json', 'r'))


class sunrgbd(torch.utils.data.Dataset):
    def __init__(self, root,mode,transform=None):

        self.root = root
        self.transform = transform

        self.images_directories = []


        if os.path.exists(root+'/testfile_dirs.pickle'):
            with open(root+'/trainfile_dirs.pickle', 'rb') as file:
                self.train = pickle.load(file)
            with open(root+'/valfile_dirs.pickle', 'rb') as file:
                self.val = pickle.load(file)
            with open(root+'/testfile_dirs.pickle', 'rb') as file:
                self.test = pickle.load(file)
        else:
            for current_dir, dirs, files in os.walk(root):
                if 'image' in dirs:
                    self.images_directories.append(current_dir)
            self.train = []
            self.test = []
            self.val = []

            for i, file in enumerate(self.images_directories):
                if i % 10 == 0:
                    self.test.append(file)
                elif i % 10 == 1:
                    self.val.append(file)
                else:
                    self.train.append(file)

            with open(root+'/testfile_dirs.pickle','wb') as file:
                pickle.dump(self.test,file)
            with open(root+'/valfile_dirs.pickle','wb') as file:
                pickle.dump(self.val,file)
            with open(root+'/trainfile_dirs.pickle','wb') as file:
                pickle.dump(self.train,file)

        if mode == 'train':
            self.filenames = self.train
        elif mode == 'test':
            self.filenames = self.test
        elif mode == 'valid':
            self.filenames = self.val
        else:
            raise ValueError('file loading error: define mode failed')

    def convertlabels(self, mask):
        for i in transfer_table.keys():
            mask = np.where(mask == float(i),transfer_table[i],mask)
        return mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        print(filename)
        print(os.path.join(filename, 'image'))
        print(self.root)
        walk_im = os.walk(os.path.join(filename, 'image')).__next__()  # returns walk list with root dir and image name
        image_path = os.path.join(walk_im[0], walk_im[2][0])
        walk_de = os.walk(os.path.join(filename, 'depth')).__next__()  # returns walk list with root dir and image name
        depth_path = os.path.join(walk_de[0], walk_de[2][0])
        mask_path = os.path.join(filename + "/mask.tif")

        image = np.array(Image.open(image_path).convert("RGB"))
        depth = np.array(Image.open(depth_path))
        mask = np.array(Image.open(mask_path))


        sample = dict(image=image, mask=mask, depth=depth)
        if self.transform is not None:
            sample = self.transform(**sample)

        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        depth = np.array(Image.fromarray(sample["depth"]).resize((256, 256), Image.NEAREST))

        mask = self.convertlabels(mask)

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["depth"] = np.expand_dims(depth, 0)

        return sample
