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
        self.label = None
        self.data = None
        self.attribute = None
