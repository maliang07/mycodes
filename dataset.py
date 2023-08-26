import os

import numpy as np
import h5py
import random
import cv2
import torch

class FE_EXT_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, images_lst, transf):
        self.image_dir = image_dir
        self.images_lst = images_lst
        self.transf = transf

    def __len__(self):
        return len(self.images_lst)

    def read_png(self, png_file):
        image = cv2.imread(png_file)
        return image

    def __getitem__(self, index):
        filename = self.images_lst[index]
        filename = os.path.join(self.image_dir, filename)
        imagename, image = self.read_png(filename)
        imagename, torch.tensor(image)


class Prost_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, indinces, filedir):
        self.df = df
        self.indinces = indinces
        self.filedir = filedir

    def __len__(self):
        return len(self.indinces)

    def read_npy(self, csv_file):
        with h5py.File(self.filedir + csv_file +'.h5', 'r') as f:
            tokens = f[csv_file]
            tokens = np.array(tokens)
            sample_list = [i for i in range(len(tokens))]
            sample_list = random.sample(sample_list, len(tokens))
            tokens = tokens[sample_list, :]

        return tokens

    def read_png(self, png_file):
        image = cv2.imread(png_file)
        return image

    def __getitem__(self, index):
        newindex = self.indinces[index]
        row = self.df.iloc[newindex]
        img_id = row.image_id
        token = self.read_npy2(img_id, self.dataset)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return torch.tensor(token), torch.tensor(label)