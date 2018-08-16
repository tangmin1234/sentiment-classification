#-*- coding: UTF-8 -*-  
import copy
import random
import logging
import os
import cPickle as pickle

from tqdm import tqdm
import numpy as np

        
class Vocab(object):
    def __init__(self, config, filename, maxn=100000):
        self.logger = logging.getLogger('{}'.format(config.model_name))
        self.config = config
        # if os.path.exists("../model/{}/vocab/voc.pkl".format(config.dataname)):
        #     self.voc = pickle.load(open("../model/{}/vocab/voc.pkl".format(config.dataname), 'r'))
        # else:

        with open(filename, 'r') as fin:
            lines = fin.readlines()[:maxn]
            lines = map(lambda x: x.decode('utf-8').split(), lines)
        self.voc = {}
        self.voc[u'pad'] = 0
        self.voc[u'unk'] = 1

        for item in tqdm(lines):
            self.add(item[0])
        
        # pickle.dump(self.voc, open("../model/{}/vocab/voc.pkl".format(config.dataname), 'w'))

        self.emb_dim = config.emb_dim
        self.embeddings = None
        self.logger.info("unfiltered vocab size = {}, including 'pad' and 'unk' token".format(self.size()))

    def get_id(self, word):
        try:
            return self.voc[word]
        except:
            return 1

    def add(self, token):
        token = token.lower() if self.config.lower else token
        if token in self.voc:
            idx = self.voc[token]
        else:
            idx = len(self.voc)
            self.voc[token] = idx
        return idx

    def size(self):
        self.voc_size = len(self.voc)
        return self.voc_size

    def randomly_init_embeddings(self, emb_dim):
        self.emb_dim = emb_dim
        self.embeddings = np.random.uniform(low=-0.25, high=0.25, size=(self.size(), self.emb_dim))
        for token in [u'pad', u'unk']:
            self.embeddings[self.get_id(token)] = np.zeros([self.emb_dim])

    def load_pretrained_embeddings(self, embedding_path):
        config = self.config
        embedding_save_path = "../model/{}/vocab/vocab_{}d.embedding".format(config.dataname, config.emb_dim)
        voc_save_path = "../model/{}/vocab/vocaburary.dict".format(config.dataname)
        if os.path.exists(embedding_save_path) and os.path.exists(voc_save_path):
            self.embeddings = np.loadtxt(embedding_save_path)
            self.emb_dim = self.embeddings.shape[1]
            self.voc = pickle.load(open(voc_save_path, 'r'))
        else:
            trained_embeddings = {}
            with open(embedding_path, 'r') as fin:
                for line in tqdm(fin):
                    contents = line.strip().split()
                    token = contents[0].decode('utf8')
                    # if token in self.voc.keys():
                    #     trained_embeddings[token] = list(map(float, contents[1:]))
                    if token not in self.voc:
                        continue
                    trained_embeddings[token] = list(map(float, contents[1:]))
                
                if self.emb_dim is None:
                    self.emb_dim = len(contents) - 1

            self.logger.info("number of token in fasttext: {}".format(len(trained_embeddings.keys())))
            # self.logger.info("number of token in glove: {}".format(len(trained_embeddings.keys())))

            self.voc = {}
            for token in [u'pad', u'unk']:
                self.add(token)
            for token in trained_embeddings.keys():
                self.add(token)

            self.embeddings = np.zeros([self.size(), self.emb_dim])

            for token in self.voc.keys():
                if token in trained_embeddings.keys():
                    self.embeddings[self.get_id(token)] = trained_embeddings[token]
            
            np.savetxt(open(embedding_save_path, 'w'), self.embeddings)
            pickle.dump(self.voc, open(voc_save_path, 'w'))

    def convert_to_ids(self, tokens):
        vec = [self.get_id(token) for token in tokens]
        return vec

def load_vocab(config):
    logger = logging.getLogger('{}'.format(config.model_name))
    if os.path.exists("../model/{}/vocab/vocab_{}d.data".format(config.dataname, config.emb_dim)):
        logger.info("loading vocab object from dumped file...")
        vocabulary = pickle.load(open("../model/{}/vocab/vocab_{}d.data".format(config.dataname, config.emb_dim), 'rb'))
    else:
        vocabulary = Vocab(config, '../data/' + config.dataname + '/wordlist.txt')
        
        if config.emb_dim != 300:
            embedding_path = os.path.join(config.embedding_path, "glove.6B.{}d.txt".format(config.emb_dim))
        else:
            if config.fasttext:
                embedding_path = os.path.join("/home/tangmin/data/word2vec/fasttext", "wiki-news-300d-1M.vec")
            else:   
                embedding_path = os.path.join(config.embedding_path, "glove.840B.300d.txt")

        if config.use_pretrain_embedding:
            logger.info("load pretrained embeddings")
            vocabulary.load_pretrained_embeddings(embedding_path)
        else:
            logger.info("load randomly initialized embeddings")
            vocabulary.randomly_init_embeddings(config.emb_dim)
        logger.info("saving vocab...")
        # with open("../model/{}/vocab/vocab_{}d.data".format(config.dataname, config.emb_dim), 'wb') as fout:
        #     pickle.dump(vocabulary, fout)
    
    return vocabulary

class UserTable(object):
    def __init__(self, filename, maxn=100000):
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        self.size = len(lines)

        self.usrlist = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]
        self.usrdict = dict(self.usrlist)
        self.size = len(self.usrdict)

    def get_id(self, usr):
        try:
            return self.usrdict[usr]
        except:
            return self.size

class ProductTable(object):
    def __init__(self, filename, maxn=100000):
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        self.size = len(lines)

        self.prdlist = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]
        self.prddict = dict(self.prdlist)
        self.size = len(self.prddict)

    def get_id(self, prd):
        try:
            return self.prddict[prd]
        except:
            return self.size