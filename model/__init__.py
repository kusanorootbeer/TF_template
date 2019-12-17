# -*- coding: utf-8 -*-
from .vae import VariationalAutoEncoder
from .cvae import ConditionalVariationalAutoEncoder
from .mlp_classifier import MLP_Classifier

_model_map = {
    "vae": VariationalAutoEncoder,
    "cvae": ConditionalVariationalAutoEncoder,
    "mlp_classifier": MLP_Classifier,
}


def get_model(model_name):
    return _model_map[model_name]


def get_model_list():
    return _model_map.keys()
