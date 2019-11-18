#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import logging
import git

import numpy as np
import tensorflow as tf

from dataset import get_dataset_list, get_dataset
from model import get_model_list, get_model


class FlexibleArgumentParser():
    def __init__(self, parser):
        self.parser = parser
        self._arguments = []

    def add_argument(self, *kargs, **kwards):
        ''' すでに同じ引数で登録されていたら無視して False 返す add_argument
            同じ引数 : dict の __eq__
        '''
        argument = (kargs, kwards)
        if self._arguments.count(argument) > 0:
            return False
        self._arguments.append(argument)
        self.parser.add_argument(*kargs, **kwards)
        return True


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=get_model_list())
    parser.add_argument("--dataset", type=str, choices=get_dataset_list())
    parser.add_argument("--out_dir", type=str, help="output image dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser


def get_logger(args):
    log_dir_name = args.out_dir + "/" + args.dataset + "/" + args.model
    ind = 0
    while(True):
        ind += 1
        if not os.path.exists(log_dir_name + str(ind)):
            os.makedirs(log_dir_name + str(ind))
            break
    log_dir_name = log_dir_name + str(ind) + "/"
    log_file_name = log_dir_name + "log.log"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.level = logging.INFO
    if log_file_name:
        filehandler = logging.FileHandler(log_file_name, mode="w")
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(filehandler)
    return logger, log_dir_name


def build_model(args, model_args, model_class, dataset):
    model_args.config = dataset.config
    model = model_class(model_args)
    return model


def main(argv=None):
    args, args_left = get_parser().parse_known_args(argv)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger, log_dir_name = get_logger(args)
    logger.info("\n" + "\n".join(["--{}={}".format(a, getattr(args, a))
                                  for a in dir(args) if not a[0] == "_"]))
    logger.info(
        "git hash: " + git.Repo(search_parent_directories=True).head.object.hexsha + "\n")
    model_class = get_model(args.model)
    model_arg_parser = argparse.ArgumentParser()
    model_class.add_argument(FlexibleArgumentParser(
        model_arg_parser))  # 引数の登録の重複を許す
    model_args, model_args_left = model_arg_parser.parse_known_args(argv)

    logger.info("create dataset")
    dataset = get_dataset(args)
    logger.info("build model")
    model = build_model(args, model_args, model_class, dataset)

    logger.info("start train")
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)))

    # import pdb
    # pdb.set_trace()
    # # TensorBoardで追跡する変数を定義
    # with tf.name_scope('summary'):
    #     tf.summary.scalar('loss', model.loss)
    #     merged = tf.summary.merge_all()
    #     writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        model.fit(dataset, logger, log_dir_name)


if __name__ == "__main__":
    main()
