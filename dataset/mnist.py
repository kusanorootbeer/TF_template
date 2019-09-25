from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from .dataset import Dataset
import numpy as np


class Mnist(Dataset):
    def __init__(self, args):
        super().__init__(args)
        mnist = input_data.read_data_sets("DATA_mnist/", one_hot=True)
        self.input_shape = np.array([1, 28, 28])
        self.input_size = 28*28
        self.train_data = mnist.train._images.reshape((-1, 28, 28))
        self.train_label = mnist.train._labels    # one hot label
        self.test_data = mnist.test._images.reshape((-1, 28, 28))
        self.test_label = mnist.test._labels    # one hot label

        self.config["input_size"] = 28*28
        self.config["input_shape"] = (1, 28, 28)
        self.config["train_itrs"] = self.train_data.shape[0] // args.batch_size
        if self.train_data.shape[0] % args.batch_size != 0:
            self.config["train_itrs"] = self.config["train_itrs"] + 1
        # self.config["test_itrs"] = self.test_data.shape[0]
