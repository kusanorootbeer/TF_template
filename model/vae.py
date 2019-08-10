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
        parser.add_argument("--units", type=str)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--out_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--epochs", type=int)

    def __init__(self, args):
        self.input_size = args.input_size
        self.units = eval(args.units)
        self.ls = args.lr
        self.out_dir = args.out_dir
        self.batch_size = args.batch_size
        self.epoch_batchs = args.epoch_batchs
        self.epochs = args.epochs

        self.act = tf.nn.relu

        self.x_base = tf.placeholder(tf.float32, [None, self.input_size])
        self.is_training = tf.placeholder(tf.bool)
        self.x_aim = tf.placeholder(tf.float32, [None, self.input_size])
        # self.dropout_prob = tf.nn.placeholder(tf.float32)
        # self.x_drop = tf.nn.dropout(self.x, rate=self.dropout_prob)

        self._build_qz_x(self.x_base)
        self._build_px_z(self.z)
        self._build_loss()
        optimizer = tf.train.AdamOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def _train_batch(self, batch):
        train_batch, teach_batch = batch
        loss, _ = tf.get_default_session().run(
            [self.loss, self.train_step],
            feed_dict={
                self.x_base: train_batch,
                self.is_training: True,
                self.x_aim: teach_batch,
            })
        return loss

    def _evaluate(self, dataset):
        test_batch = core.common.get_test_batch(dataset)
        loss, no_text_batch = tf.get_default_session().run(
            [self.loss, self.x_rec],
            feed_dict={
                self.x_base: test_batch,
                self.is_training: False,
            })
        return loss, no_text_batch


    def _build_qz_x(self, x):
        self.qz = [x]
        for i, unit in enumerate(self.units[:-1]):
            hz_x = core.common.dense_layer(self.qz[-1], unit, use_bias=True)
            hz_x = core.common.batch_normalization(hz_x, self.is_training)
            hz_x = self.act(hz_x)
            self.qz.append(hz_x)

        # N(mean, log_sigma_sq)
        self.qz_mean = dense_layer(self.qz[-1], self.units[-1], use_bias=True)
        self.qz_log_sigma_sq = dense_layer(self.qz[-1], self.units[-1], use_bias=True)
        # sample
        self.z, self.qz_eps = core.common.random_sample(
            self.qz_mean,
            self.qz_log_sigma_sq,
            self.is_training,
        )

    def _build_px_z(self, z):
        self.px = [z]
        for i, units in enumerate(reversed(self.units[:-1])):
            hx_z = core.common.dense_layer(self.px)
            hx_z = core.common.batch_normalization(hx_z, self.is_training)
            hx_z = self.act(hx_Z)
            self.px.append(hx_Z)

        # N(mean, log_sigma_sq)
        self.px_mean = dense_layer(self.px[-1], self.input_size, use_bias=True)
        self.px_log_sigma_sq = dense_layer(self.px[-1], self.input_size, use_bias=True)
        # sample
        self.x_rec, self.px_eps = random_sample(
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
            for itr in range(self.epoch_batchs):
                batch = core.common.make_train_batch(dataset, itr, self.epoch_batchs)
                loss = self._train_batch(batch)
                logger.info("epoch:{:5}itr:{:5}loss:{}".format(epoch, itr, loss))
            loss, out_images = self._evaluate(dataset)
            logger.info("epoch:{:5}loss:{}")
            out_image = out_images[0]
            plt.figure(figsize=(40, 40))
            plt.imshow(out_image.reshape(dataset.shape))
            plt.savefig("{}/{}.png".format(self.out_dir, epoch))
            plt.close()






