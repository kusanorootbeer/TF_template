# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import random
from pathlib import Path
from PIL import Image


class Dataset():
    def __init__(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def shuffle(self):
        np.random.shuffle(self.train_data_indices)
