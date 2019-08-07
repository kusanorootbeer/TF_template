# -*- coding: utf-8 -*-
from .mnist import Mnist
#from .hoge import Hoge

_dataset_map = {
    "mnist": Mnist,
    #"hoge": Hoge,

}


def get_dataset(dataset_name, dataroot):
    return _dataset_map[dataset_name](dataroot)


def get_dataset_list():
    return _dataset_map.keys()
