# -*- coding: utf-8 -*-
from .mnist import Mnist
#from .hoge import Hoge

_dataset_map = {
    "mnist": Mnist,
    #"hoge": Hoge,

}


def get_dataset(args):
    return _dataset_map[args.dataset](args)


def get_dataset_list():
    return _dataset_map.keys()
