# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.stats
import tensorflow as tf


def build_dense_layers(first_input, units_list, use_bias_list, activation_list, normalization_list, is_training=True):
    """ argument example
    first_input: (batch_size, input_size)
    units_list: [200,100,10]
    use_bias_list: [True, True, False]
    activation_list: [tf.nn.relu, tf.nn.relu, None]
    normalization_list: ["batch", "layer", None]
    """
    hidden = first_input
    for unit, use_bias, actiavtion, normalization in zip(units_list, use_bias_list, activation_list, normalization_list):
        hidden = dense_layer(
            hidden, unit, use_bias=use_bias, activation=actiavtion)
        hidden = normalize(hidden, normalization, is_training)
    return hidden


def dense_layer(inputs, units, use_bias, std=0.02, activation=None):
    ''' build dense layer with custom initialization '''
    return tf.layers.dense(
        inputs,
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(stddev=std),
        bias_initializer=tf.zeros_initializer(),
    )


def normalize(x, normalization_code, is_training):
    if normalization_code == None:
        return x
    elif normalization_code == "batch":
        return batch_normalization(x, is_training)
    elif normalization_coda == "layer":
        return layer_normalization(x)


def layer_normalization(x, EPS=1e-8):
    assert len(x.get_shape()) == 2
    x_mean, x_variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    x_std = tf.sqrt(x_variance)
    return (x - x_mean)*1.0 / (x_std + EPS)


def batch_normalization(x, is_training, decay=0.9, eps=1e-5):
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]
    beta = tf.Variable(tf.zeros([n_out]))
    gamma = tf.Variable(tf.ones([n_out]))
    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(is_training, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def random_sample(mean, log_sigma_sq, is_training):
    ''' sample from N(mean, log_sigma_sq), if is_training '''
    shape = tf.shape(log_sigma_sq)
    eps = tf.random_normal(shape, dtype=tf.float32)
    # inject noise only when tranining
    eps = tf.cond(is_training, lambda: eps, lambda: tf.zeros(shape))
    z = mean + tf.multiply(tf.exp(log_sigma_sq * 0.5), eps)
    return z, eps


def gaussian_nll(x, mean, ln_var, INF=1e3, EPS=-1e3):
    #EPS = 1e-8
    #INF = math.log(10)
    ln_var = tf.clip_by_value(ln_var, EPS, INF)
    x_prec = tf.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec
    return (ln_var + math.log(2 * math.pi) + x_power) / 2
