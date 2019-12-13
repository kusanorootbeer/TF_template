# -*- coding: utf-8 -*-

from . import core
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')


class VariationalAutoEncoder():
    @classmethod
    def add_argument(cls, parser):
        parser.add_argument("--input_size", type=int)
        # parser.add_argument("--input_shape", type=str)
        parser.add_argument("--units", type=str)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--out_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--epochs", type=int)

    def __init__(self, args):
        """ this model use config:
        # input_shape: input image data shape [-1, height, width, channel]
        input_size: input image data size [height * width * channel]
        train_itrs: number of training for a epoch [number_of_train_data / batch_size]
        data_shape: image data shape [height, width, channel=3] or [height, width] 
        """

        for keys in args.config.keys():
            exec("self.{} = {}".format(keys, args.config[keys]))

        self.units = eval(args.units)
        self.lr = args.lr
        self.out_dir = args.out_dir
        self.batch_size = args.batch_size
        self.epochs = args.epochs

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

    def _train(self, batch):
        train_batch = batch
        loss, _ = tf.get_default_session().run(
            [self.loss, self.train_step],
            feed_dict={
                self.x: train_batch,
                self.is_training: True,
            })
        return loss

    def _evaluate(self, batch):
        loss, out_batch = tf.get_default_session().run(
            [self.loss, self.x_rec],
            feed_dict={
                self.x: batch,
                self.is_training: False,
            })
        return loss, out_batch

    def _full_evaluate(self, dataset):
        loops, rest = (dataset.test_num//self.batch_size,
                       dataset.test_num % self.batch_size)
        losses = np.zeros(loops+1)
        for i in range(loops+1):
            ind = i * self.batch_size
            indices = np.arange(ind, ind+self.batch_size)
            if i == loops:
                indices = np.arange(ind, ind+rest)
            batch = dataset.get_batch(
                {"region": "test", "batch": "data", "indices": indices})
            loss, _ = self._evaluate(batch)
            losses[i] = loss
        return loss

    def _build_qz_x(self, x):
        # define hidden layers parameters
        hidden_layers_num = self.units[:-1].__len__()
        use_bias_list = [True] * hidden_layers_num
        activation_list = [self.act] * hidden_layers_num
        normalization_list = ["batch"] * hidden_layers_num

        last_hidden = core.common.build_dense_layers(
            x, self.units[:-1], use_bias_list, activation_list, normalization_list, self.is_training)

        # N(mean, log_variance)
        self.qz_mean = core.common.dense_layer(
            last_hidden, self.units[-1], use_bias=True)
        self.qz_log_variance = core.common.dense_layer(
            last_hidden, self.units[-1], use_bias=True)

        # sample
        self.z, self.qz_eps = core.common.random_sample(
            self.qz_mean,
            self.qz_log_variance,
            self.is_training,
        )

    def _build_px_z(self, z):
        # define hidden layers parameters
        hidden_layers_num = self.units[:-1].__len__()
        use_bias_list = [True] * hidden_layers_num
        activation_list = [self.act] * hidden_layers_num
        normalization_list = ["batch"] * hidden_layers_num

        last_hidden = core.common.build_dense_layers(
            z, reversed(self.units[:-1]), use_bias_list, activation_list, normalization_list, self.is_training)

        # N(mean, log_variance)
        self.px_mean = core.common.dense_layer(
            last_hidden, self.input_size, use_bias=True)
        self.px_log_variance = core.common.dense_layer(
            last_hidden, self.input_size, use_bias=True)
        # sample
        self.x_rec, self.px_eps = core.common.random_sample(
            self.px_mean,
            self.px_log_variance,
            self.is_training,
        )

    def _build_loss(self):
        self.nll_qz = core.common.gaussian_nll(
            self.z, self.qz_mean, self.qz_log_variance)
        self.nll_pz = core.common.gaussian_nll(
            self.z, tf.zeros_like(self.z), tf.zeros_like(self.z))
        self.nll_px = core.common.gaussian_nll(
            self.x, self.px_mean, self.px_log_variance)

        self.loss_z = -tf.reduce_mean(self.nll_qz) + \
            tf.reduce_mean(self.nll_pz)
        self.loss_x = tf.reduce_mean(self.nll_px)

        self.loss = self.loss_z + self.loss_x

    def fit(self, dataset, logger, log_dir_name):
        for epoch in range(1, self.epochs+1):
            for itr in range(self.train_itrs):
                batch = dataset.get_batch(
                    {"itr": itr, "region": "train", "batch": "data"})
                loss = self._train(batch)
                logger.info("epoch:{:5}  itr:{:5}  loss:{}"
                            .format(epoch, itr+1, loss))
            batch = dataset.get_batch(
                {"region": "test", "batch": "data"})
            loss, out_images = self._evaluate(batch)
            logger.info("epoch:{:5}  loss:{}".format(epoch, loss))
            dataset.shuffle()
        loss = self._full_evaluate(dataset)
        logger.info("\n Finally, Full test data loss: {:5}".format(loss))

        dataset.save_fig(fig_name=log_dir_name+"batch",
                         data_array=batch[0].reshape(self.data_shape))
        dataset.save_fig(fig_name=log_dir_name+"out",
                         data_array=out_images[0].reshape(self.data_shape))

    def assess(self, dataset, logger, log_file_name):
        loss_list = []
        x_mean_list = []
        x_log_variance_list = []

        for data_index in range(dataset.test_data.shape[0]):
            loss, out_x_mean, out_x_log_variance = tf.get_default_session().run(
                [self.loss, self.px_mean, self.px_log_variance],
                feed_dict={
                    self.x: dataset.test_data[data_index],
                    self.is_training: False,
                })
            loss_list.append(loss)
            x_mean_list.append(out_x_mean)
            x_log_variance_list.append(out_x_log_variance)
        losses = np.array(loss_list)
        x_means = np.array(x_mean_list)
        x_log_variances = np.array(x_log_variance_list)
        save_path = log_file_name.split(".")[0]
        np.save(save_path + "_loss.npy", losses)
        np.save(save_path + "_mean.npy", x_means)
        np.save(save_path + "_log_variance.npy", x_log_variances)
