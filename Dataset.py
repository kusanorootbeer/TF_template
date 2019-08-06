# -*- coding: utf-8 -*-
import csv
import os
from functools import partial
import glob
import numpy as np
import random
from pathlib import Path
import cv2 as cv
import math


class Dataset():
    def __init__(self, args):
        self.data_root = args.data_root
        self.train_data_dir = args.train_data_dir
        self.image_nest = args.image_nest
        self.epoch_batchs = args.epoch_batchs

    def load(self):
        pass

