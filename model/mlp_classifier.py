# -*- coding: utf-8 -*-

from . import core
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')


class MLP_Classifier():
    @classmethod
    def add_argument(cls, parser):
        parser.add_argument("--input_size", type=int)
        parser.add_argument("--label_size", type=int)
        parser.add_argument("--units", type=str)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--out_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--epochs", type=int)

    def __init__(self, args):
        """ this model use config:
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
        self.y = tf.placeholder(tf.float32, [None, self.label_size])
        self.is_training = tf.placeholder(tf.bool)

        self._build_classifier(self.x)
        self._build_loss()
        optimizer = tf.train.AdamOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def _train(self, batch):
        data_batch, label_batch = batch
        loss, prediction, _ = tf.get_default_session().run(
            [self.loss, self.prediction, self.train_step],
            feed_dict={
                self.x: data_batch,
                self.y: label_batch,
                self.is_training: True,
            })
        acc = core.common.calc_acc(prediction, label_batch)
        return loss, acc

    def _evaluate(self, batch, opt={}):
        data_batch, label_batch = batch
        loss, prediction = tf.get_default_session().run(
            [self.loss, self.prediction],
            feed_dict={
                self.x: data_batch,
                self.y: label_batch,
                self.is_training: False,
            })
        if opt.get("full"):
            return loss, prediction
        acc = core.common.calc_acc(prediction, label_batch)
        return loss, acc

    def _full_evaluate(self, dataset):
        loops, rest = (dataset.test_num//self.batch_size,
                       dataset.test_num % self.batch_size)
        labels = np.zeros((dataset.test_num, self.label_size))
        predictions = np.zeros_like(labels)
        for i in range(loops+1):
            ind = i * self.batch_size
            indices = np.arange(ind, ind+self.batch_size)
            if i == loops:
                indices = np.arange(ind, ind+rest)
            data_batch, label_batch = dataset.get_batch(
                {"region": "test", "batch": "data+label", "indices": indices})
            _, prediction = self._evaluate(
                batch=(data_batch, label_batch), opt={"full": True})
            labels[indices] = label_batch
            predictions[indices] = prediction
        return core.common.calc_acc(predictions, labels)

    def _build_classifier(self, x):
        # define hidden layers parameters
        hidden_layers_num = self.units[:-1].__len__()
        use_bias_list = [True] * hidden_layers_num
        activation_list = [self.act] * hidden_layers_num
        normalization_list = ["batch"] * hidden_layers_num

        last_hidden = core.common.build_dense_layers(
            x, self.units[:-1], use_bias_list, activation_list, normalization_list, self.is_training)

        self.out_layer = core.common.dense_layer(
            last_hidden, self.label_size, use_bias=True, activation=tf.nn.softmax)
        self.prediction = tf.one_hot(
            tf.argmax(self.out_layer, axis=1), depth=self.label_size)

    def _build_loss(self):
        self.loss_classify = - self.y * self.out_layer
        self.loss = tf.reduce_mean(self.loss_classify)

    def fit(self, dataset, logger, log_dir_name):
        for epoch in range(1, self.epochs+1):
            for itr in range(self.train_itrs):
                batch = dataset.get_batch(
                    {"itr": itr, "region": "train", "batch": "data+label"})
                loss, acc = self._train(batch)
                logger.info("epoch:{:5}  itr:{:5}  loss:{:5}  acc:{:5}"
                            .format(epoch, itr+1, loss, acc))
            batch = dataset.get_batch(
                {"region": "test", "batch": "data+label"})
            loss, prediction = self._evaluate(batch)
            logger.info(
                "epoch:{:5}  loss:{:5}  acc{:5}".format(epoch, loss, acc))
            dataset.shuffle()
        acc = self._full_evaluate(dataset)
        print("\n")
        logger.info(" Finally, Full test data accuracy: {:5}".format(acc))
