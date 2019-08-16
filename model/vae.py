# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from . import core

class VariationalAutoEncoder():
    @classmethod
    def add_argument(cls, parser):
        parser.add_argument("--input_size", type=int)
        parser.add_argument("--input_shape", type=str)
        parser.add_argument("--units", type=str)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--out_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--net_type", type=str, choices=["MLP", "CNN", "MLP_CNN"], default="MLP")

    def __init__(self, args):
        """ this model use config:
        input_shape: input image data shape [channel, width, height]
        input_size: input image data size [channel * width * height]
        train_itrs: number of training for a epoch [number_of_train_data / batch_size]
        """
        for keys in args.config.keys():
            exec("self.{} = {}".format(keys, args.config[keys]))

        self.units = eval(args.units)
        self.lr = args.lr
        self.out_dir = args.out_dir
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.net_type = args.net_type

        self.act = tf.nn.relu
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.is_training = tf.placeholder(tf.bool)

        self._build_qz_x(self.x)
        self._build_px_z(self.z)
        self._build_loss()
        optimizer = tf.train.AdamOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def _train_batch(self, batch):
        train_batch = batch
        loss, _ = tf.get_default_session().run(
            [self.loss, self.train_step],
            feed_dict={
                self.x: train_batch,
                self.is_training: True,
            })
        return loss

    def _evaluate(self, dataset):
        test_batch = core.common.get_test_batch(dataset)
        loss, out_batch = tf.get_default_session().run(
            [self.loss, self.x_rec],
            feed_dict={
                self.x: test_batch,
                self.is_training: False,
            })
        return loss, out_batch


    def _build_qz_x(self, x):
        self.qz = [x]
        for i, unit in enumerate(self.units[:-1]):
            hz_x = core.common.dense_layer(self.qz[-1], unit, use_bias=True)
            hz_x = core.common.batch_normalization(hz_x, self.is_training)
            hz_x = self.act(hz_x)
            self.qz.append(hz_x)

        # N(mean, log_sigma_sq)
        self.qz_mean = core.common.dense_layer(self.qz[-1], self.units[-1], use_bias=True)
        self.qz_log_sigma_sq = core.common.dense_layer(self.qz[-1], self.units[-1], use_bias=True)
        # sample
        self.z, self.qz_eps = core.common.random_sample(
            self.qz_mean,
            self.qz_log_sigma_sq,
            self.is_training,
        )

    def _build_px_z(self, z):
        self.px = [z]
        for i, unit in enumerate(reversed(self.units[:-1])):
            hx_z = core.common.dense_layer(self.px[-1], unit, use_bias=True)
            hx_z = core.common.batch_normalization(hx_z, self.is_training)
            hx_z = self.act(hx_z)
            self.px.append(hx_z)

        # N(mean, log_sigma_sq)
        self.px_mean = core.common.dense_layer(self.px[-1], self.input_size, use_bias=True)
        self.px_log_sigma_sq = core.common.dense_layer(self.px[-1], self.input_size, use_bias=True)
        # sample
        self.x_rec, self.px_eps = core.common.random_sample(
            self.px_mean,
            self.px_log_sigma_sq,
            self.is_training,
        )

    def _build_loss(self):
        self.nll_qz = core.common.gaussian_nll(self.z, self.qz_mean, self.qz_log_sigma_sq)
        self.nll_pz = core.common.gaussian_nll(self.z, tf.zeros_like(self.z), tf.zeros_like(self.z))
        self.nll_px = core.common.gaussian_nll(self.x_rec, self.px_mean, self.px_log_sigma_sq)

        self.loss_z = -tf.reduce_mean(self.nll_qz) + tf.reduce_mean(self.nll_pz)
        self.loss_x = tf.reduce_mean(self.nll_px)

        self.loss = self.loss_z + self.loss_x

    def fit(self, dataset, logger):
        for epoch in range(1, self.epochs+1):
            for itr in range(self.train_itrs):
                batch = self.make_train_batch(dataset, itr)
                loss = self._train_batch(batch)
                logger.info("epoch:{:5}itr:{:5}loss:{}".format(epoch, itr, loss))
            loss, out_images = self._evaluate(dataset)
            logger.info("epoch:{:5}loss:{}")
            out_image = out_images[0]
            plt.figure(figsize=(40, 40))
            plt.imshow(out_image.reshape(dataset.shape))
            plt.savefig("{}/{}.png".format(self.out_dir, epoch))
            plt.close()

    def make_train_batch(self, dataset, itr):
        index = itr * self.batch_size
        train_data_batch = dataset.train_data[index:index+self.batch_size]
        if self.net_type == "MLP":
            train_data_batch = train_data_batch.reshape(-1, self.input_size)
        return train_data_batch







