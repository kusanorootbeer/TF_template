# -*- coding: utf-8 -*-
from .vae import VariationalAutoEncoder

_model_map = {
    "vae": VariationalAutoEncoder,
}


def get_model(model_name):
    return _model_map[model_name]


def get_model_list():
    return _model_map.keys()
