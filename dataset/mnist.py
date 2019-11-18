from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from .dataset_image_label import Dataset_image_label
import numpy as np


class Mnist(Dataset_image_label):
    def __init__(self, args):
        super().__init__(args)
        mnist = input_data.read_data_sets("DATA_mnist/", one_hot=True)
        self.train_image = mnist.train._images.reshape((-1, 28 * 28))
        self.train_label = mnist.train._labels    # one hot label
        self.test_image = mnist.test._images.reshape((-1, 28 * 28))
        self.test_label = mnist.test._labels    # one hot label
        self.batch_size = args.batch_size

        self.config["input_size"] = 28*28
        self.config["train_itrs"] = self.train_image.shape[0] // self.batch_size
        self.config["data_shape"] = (28, 28)
        if self.train_image.shape[0] % args.batch_size != 0:
            self.config["train_itrs"] = self.config["train_itrs"] + 1
        # self.config["test_itrs"] = self.test_data.shape[0]

        self.train_data_indices = np.arange(self.train_image.__len__())
