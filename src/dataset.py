#-*- coding: UTF-8 -*-  
import copy
import random
import logging
import os
import cPickle as pickle

from tqdm import tqdm
import numpy as np
     
        
def preprocess(docsbatch):
    bs = len(docsbatch)
    num_sents = [len(doc) for doc in docsbatch]
    max_sent_num = np.max(num_sents)
    length = [[len(sentence) for sentence in doc] for doc in docsbatch]
    max_len = max([max(doc_length) for doc_length in length])

    docs = np.zeros((bs, max_sent_num, max_len)).astype('int32') # [N, M, JX]
    wordmask = np.zeros((bs, max_sent_num, max_len), dtype='bool') # [N, M, JX]
    sentencemask = np.zeros((bs, max_sent_num), dtype='bool') # [N, M]
    docs_length = np.zeros((bs, max_sent_num))

    for idx, doc in enumerate(docsbatch):
        for sent_idx, sent in enumerate(doc):
            docs[idx, sent_idx, :length[idx][sent_idx]] = sent
            wordmask[idx, sent_idx, :length[idx][sent_idx]] = [True] * length[idx][sent_idx]
            sentencemask[idx, sent_idx] = True
            docs_length[idx, sent_idx] = length[idx][sent_idx]

    return docs, wordmask, sentencemask


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, config, filename, vocab, usrdict=None, prddict=None, set_name=None):
        super(Dataset, self).__init__()
        self.config = config
        if self.config.dataname == 'IMDB':
            self.config.max_sent_num = 40
            self.config.max_sent_len = 100
        self.set_name = set_name
        self.data = self.read_data(filename)
        self.data = self.convert_to_ids(vocab)
        
    def read_data(self, filename):
        config = self.config
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())
        label = np.asarray(map(lambda x: int(x[2])-1, lines), dtype=np.int32)

        if config.debug:
            lines = lines[0: 128]
            label = label[0: 128]

        if config.use_user_info and config.use_product_info:
            assert usrdict is not None and prddict is not None
            usr = map(lambda line: usrdict.get_id(line[0]), lines) 
            prd = map(lambda line: prddict.get_id(line[1]), lines)

        docs = map(lambda x: x[3][0 : len(x[3]) - 1], lines) 
        docs = map(lambda x: x.split('<sssss>'), docs)
        docs = map(lambda doc: map(lambda sentence: sentence.split(' '), doc), docs)

        sentencenum = map(lambda x : len(x), docs)
        length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)

        if config.use_user_info and config.use_product_info:
            data = zip(docs, label, usr, prd)
        else:
            data = zip(docs, label)

        return data

    def convert_to_ids(self, vocab):
        config = self.config
        data_set = self.data
        new_data_set = []
        if config.use_user_info and config.use_product_info:
            for sample in data_set:
                d_ids = []
                for s in sample[0]:
                    d_ids.append(vocab.convert_to_ids(s))
                new_data_set += [(d_ids, sample[1], sample[2], sample[3])]
        else:
            num_filtered = 0
            for sample in data_set:
                if self.set_name == 'train':
                    if len(sample[0]) > config.max_sent_num or any([len(s) > config.max_sent_len for s in sample[0]]):
                        num_filtered += 1
                        continue
                if self.set_name == 'test':
                    if len(sample[0]) > 2 * config.max_sent_num or any([len(s) > 2 * config.max_sent_len for s in sample[0]]):
                        num_filtered += 1
                        continue
                d_ids = []
                for s in sample[0]:
                    d_ids.append(vocab.convert_to_ids(s))
                new_data_set += [(d_ids, sample[1])]
            print("filtered {} sample...".format(num_filtered))
        return new_data_set


    def one_mini_batch(self, data, indices):
        config = self.config
        mb_doc = [data[i][0] for i in indices]
        mb_label = [data[i][1] for i in indices]

        if config.use_user_info and config.use_product_info:
            mb_usr = [data[i][2] for i in indices]
            mb_prd = [data[i][3] for i in indices]
        
        mb_doc, mb_wordmask, mb_sentencemask = preprocess(mb_doc)
        mb_label = np.array(mb_label)

        if config.use_user_info and config.use_product_info:
            batch_data = (mb_doc, mb_label, mb_wordmask, mb_sentencemask, mb_usr, mb_prd)
        else:
            batch_data = (mb_doc, mb_label, mb_wordmask, mb_sentencemask)

        return batch_data

    def gen_mini_batch(self, batch_size, shuffle=True):
        data = self.data
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self.one_mini_batch(data, batch_indices)

def load_dataset(config, voc, usrdict, prddict):
    logger = logging.getLogger('{}'.format(config.model_name))
    logger.info("load data...")

    trainset_dump_file = "../model/{}/data/trainset.data".format(config.dataname)
    devset_dump_file = "../model/{}/data/devset.data".format(config.dataname)
    testset_dump_file = "../model/{}/data/testset.data".format(config.dataname)

    if os.path.exists(trainset_dump_file) & os.path.exists(devset_dump_file) & os.path.exists(testset_dump_file):
        logger.info("loading dataset from dumped file...")
        trainset = pickle.load(open(trainset_dump_file, 'rb'))
        devset = pickle.load(open(devset_dump_file, 'rb'))
        testset = pickle.load(open(testset_dump_file, 'rb'))
    else:
        train_file = '../data/' + config.dataname + '/train.txt'
        dev_file = '../data/' + config.dataname + '/dev.txt'
        test_file = '../data/' + config.dataname + '/test.txt'
        trainset = Dataset(config, train_file, voc, usrdict=usrdict, prddict=prddict, set_name='train')
        devset = Dataset(config, dev_file, voc, usrdict=usrdict, prddict=prddict)
        testset = Dataset(config, test_file, voc, usrdict=usrdict, prddict=prddict)
        logger.info("trainset size = {}, devset size = {}, testset size = {}".format(len(trainset.data), len(devset.data), len(testset.data)))
        # pickle.dump(trainset, open(trainset_dump_file, 'wb'))
        # pickle.dump(devset, open(devset_dump_file, 'wb'))
        # pickle.dump(testset, open(testset_dump_file, 'wb'))
    
    logger.info('data loading done.')
    return trainset, devset, testset