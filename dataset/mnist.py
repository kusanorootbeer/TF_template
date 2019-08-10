from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from .dataset import Dataset

class Mnist(Dataset):
    def __init__(self, args):
        super().__init__(args)
        mnist = input_data.read_data_sets("DATA_mnist/", one_hot=True)
        self.train_data = mnist.train._images.reshape((-1, 28, 28))
        self.train_label = mnist.train._labels    # one hot label
        self.test_data = mnist.test._images.reshape((-1, 28, 28))
        self.test_label = mnist.test._labels    # one hot label


