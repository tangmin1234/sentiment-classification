#-*- coding: UTF-8 -*-
import sys
import os

import tensorflow as tf
import numpy as np

from config import get_args, get_logger
from dataset import Dataset, load_dataset
from vocab import Vocab, UserTable, ProductTable, load_vocab
from reinforced_network import ReinforcedNetwork as Model

def main():
    # import ipdb; ipdb.set_trace()
    config = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logger = get_logger(config)
    voc = load_vocab(config)
    config.n_voc = len(voc.voc)
    logger.info(config)
    config.embeddings = voc.embeddings

    config.use_product_info = False
    config.use_user_info = False
    if config.use_user_info and config.use_product_info:
        usrdict = UserTable('../data/' + config.dataname + '/usrlist.txt')
        prddict = ProductTable('../data/' + config.dataname + '/prdlist.txt')
        config.n_users = usrdict.size + 1
        config.n_products = prddict.size + 1
    else:
        usrdict = None
        prddict = None

    logger.info("build model...")
    with tf.device("/device:{}:{}".format(config.device_type, config.gpu_id)):
        model = Model(config)
    
    logger.info("creating session...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_allocate_rate)
    gpu_options.allow_growth = True
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=session_config)
    model.init_variable(sess)
    
    trainset, devset, testset = load_dataset(config, voc, usrdict, prddict)

    if config.load_model:
        logger.info("restoring model...")
        model.restore(sess)

    if not config.test_only:
    
        logger.info("starting training...")
        model.train(sess, trainset, devset, testset)
        logger.info("training done.")

    logger.info("starting testing...")
    test_acc, test_mae, test_rmse = model.evaluate(sess, testset)
    logger.info("final result of testset: acc = {:.4f}, mae = {:.4f}, rmse = {:.4f}".format(test_acc, test_mae, test_rmse))
    logger.info("testing done.")

if __name__ == '__main__':
    main()