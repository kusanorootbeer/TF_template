# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import random
from pathlib import Path
from PIL import Image
from .dataset import Dataset


class Dataset_image_label(Dataset):
    def __init__(self, args):
        self.train_label = None
        self.train_image = None

        self.test_image = None
        self.test_label = None
        self.test_num = None

        self.batch_size = None

        self.config = {}    # config追加用dict

    def _get_train_batch(self, option_dict={}):
        itr = option_dict.get("itr")
        index = itr * self.batch_size
        if index + self.batch_size > self.train_image.__len__():
            train_batch_indices = self.train_data_indices[index:-1]
        else:
            train_batch_indices = self.train_data_indices[index:index+self.batch_size]

        if "label" in option_dict.get("batch"):
            batch_image = self.train_image[train_batch_indices]
            batch_label = self.train_label[train_batch_indices]
            return batch_image, batch_label
        else:
            batch_image = self.train_image[train_batch_indices]
            return batch_image

    def _get_test_batch(self, option_dict={}):
        indices = np.random.randint(
            low=0, high=self.test_image.__len__(), size=self.batch_size)
        if "label" in option_dict.get("batch"):
            batch_image = self.test_image[indices]
            batch_label = self.test_label[indices]
            return batch_image, batch_label
        else:
            batch_image = self.test_image[indices]
            return batch_image

    def shuffle(self):
        np.random.shuffle(self.train_data_indices)
