# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import random
from pathlib import Path


class Dataset():
    def __init__(self, args):
        self.train_label = None
        self.train_data = None
        self.train_attribute = None

        self.test_label = None
        self.test_data = None
        self.test_attribute = None

    def normalize(self, args):
        raise NotImplementedError
