from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from .dataset_image_label import Dataset_image_label
import numpy as np
import pdb


class Mnist(Dataset_image_label):
    def __init__(self, args):
        super().__init__(args)
        mnist = input_data.read_data_sets("DATA_mnist/", one_hot=False)
        active_indices = np.empty(0, dtype=int)
        for i in range(10):
            indices = np.where(mnist.train._labels == i)[0]
            if args.dataset_limit != None:
                indices = np.random.choice(
                    indices, args.dataset_limit, replace=False)
            active_indices = np.concatenate([active_indices, indices], axis=0)
        active_indices = np.sort(active_indices)
        pdb.set_trace()
        mnist = input_data.read_data_sets("DATA_mnist/", one_hot=True)
        self.train_image = mnist.train._images[active_indices]
        self.train_label = mnist.train._labels[active_indices]
        self.test_image = mnist.test._images
        self.test_label = mnist.test._labels
        self.batch_size = args.batch_size
        self.test_num = len(self.test_label)
        pdb.set_trace()

        self.config["input_size"] = 28*28
        self.config["train_itrs"] = self.train_image.shape[0] // self.batch_size
        self.config["data_shape"] = (28, 28)
        if self.train_image.shape[0] % args.batch_size != 0:
            self.config["train_itrs"] = self.config["train_itrs"] + 1
        # self.config["test_itrs"] = self.test_data.shape[0]

        self.train_data_indices = np.arange(self.train_image.__len__())
