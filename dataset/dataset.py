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
        data_array = (255 * data_array).astype(np.uint8)
        pil_image = Image.fromarray(data_array)
        pll_image.save("{}.png".format(fig_name))
