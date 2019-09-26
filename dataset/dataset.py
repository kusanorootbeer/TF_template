# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import random
from pathlib import Path
from PIL import Image


class Dataset():
    def __init__(self, args):
        self.train_label = None
        self.train_data = None
        self.train_attribute = None

        self.test_label = None
        self.test_data = None
        self.test_attribute = None

        self.config = {}    # config追加用dict

    def normalize(self, args):
        raise NotImplementedError

    def save_fig(self, fig_name, data_array, option_dict={}):
        # if gray : data shape [height, weight]
        # if color: data shape [height, weight, 3]
        # data range 0~255 uint8
        if option_dict.get("color") == "gray":
            assert data_array.shape.__len__() == 3, "this array is not color image array"
            data_array = np.mean(data_array, axis=-1)

        data_array = (255 * data_array).astype(np.uint8)
        pil_image = Image.fromarray(data_array)
        pil_image.save("{}.png".format(fig_name))

    def get_batch(self, option_dict={}):
        if option_dict.get("region") == "train":
            return self._get_train_batch(option_dict)
        elif option_dict.get("region") == "test":
            return self._get_test_batch(option_dict)

    def _get_train_batch(self, option_dict={}):
        itr = option_dict.get("itr")
        index = itr * self.batch_size
        if index + self.batch_size > self.train_data.__len__():
            train_batch_indices = self.train_data_indices[index:-1]
        else:
            train_batch_indices = self.train_data_indices[index:index+self.batch_size]
        batch_data = self.train_data[train_batch_indices]
        batch_label = self.train_label[train_batch_indices]
        batch_attribute = self.train_attribute[train_batch_indices]
        return batch_data, batch_label, batch_attribute

    def _get_test_batch(self, option_dict={}):
        indices = np.random.randint(
            low=0, high=self.test_data.__len__(), size=self.batch_size)
        batch_data = self.test_data[indices]
        batch_label = self.test_label[indices]
        batch_attribute = self.test_attribute[indices]
        return batch_data, batch_label, batch_attribute

    def shuffle(self):
        np.random.shuffle(self.train_data_indices)
