# -*- coding: utf-8 -*-
# from .hoge import Hoge

_model_map = {
#    "hoge": Hoge,
}


def get_model(model_name):
    return _model_map[model_name]


def get_model_list():
    return _model_map.keys()
