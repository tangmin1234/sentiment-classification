from __future__ import division
import logging
import math
import numpy as np
import tensorflow as tf

from tf_utils import get_optimizer, F
use_cudnn_rnn = False
if use_cudnn_rnn:
    from tf_utils import cudnn_rnn as rnn
else:
    from tf_utils import rnn
from general import get_initializer
from nn import linear, softsel


class ReinforcedNetwork(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('{}'.format(config.model_name))
        self.global_step = tf.get_variable('global_step', [], dtype='int32', trainable=False, 
                                            initializer=tf.constant_initializer(0))
        self.add_placeholder()
        self.build_network()
        self.build_objective()
        self.build_metrics()
        self.build_optimizer()

        self.saver = tf.train.Saver(max_to_keep=1)
        if config.summary:
            summary_path = "../model/{}/log/".format(config.dataname)
            self.writer = tf.summary.FileWriter(summary_path)
        self.logger.info("finished initialize network.")

    def add_placeholder(self):
        self.docs = tf.placeholder(tf.int32, [None, None, None], 'docs') # [N, M, JX]
        self.label = tf.placeholder(tf.int32, [None], 'label') # [N]
        
        self.wordmask = tf.placeholder(tf.bool, [None, None, None], 'wordmask') # [N, M, JX]
        self.sentencemask = tf.placeholder(tf.bool, [None, None], 'sentencemask') # [N, M]
        self.is_train = tf.placeholder(tf.bool, [], 'is_train')
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        if self.config.use_user_info and self.config.use_product_info:
            self.usr = tf.placeholder(tf.int32, [None], 'usr')
            self.prd = tf.placeholder(tf.int32, [None], 'prd')

    def build_network(self):
        config = self.config

        if config.debug:
            self.tensor_dict = {}

        VW, dw, d = config.n_voc, config.emb_dim, config.hidden_dim
        N, M, JX = tf.shape(self.docs)[0], tf.shape(self.docs)[1], tf.shape(self.docs)[2]
        
        with tf.variable_scope("emb"):
            with tf.variable_scope('emb_var'), tf.device('/cpu:0'):
                word_emb_mat = tf.get_variable('word_emb_mat', shape=[VW, dw], dtype='float', 
                                                initializer=get_initializer(config.embeddings), 
                                                trainable=config.tune_embedding)
                docs_emb = tf.nn.embedding_lookup(word_emb_mat, self.docs) # [N, M, JX, d]

        if config.debug:
            self.tensor_dict['docs_emb'] = docs_emb
                
        wordmask = tf.reshape(self.wordmask, [N*M, JX])
        length = tf.reduce_sum(tf.to_int32(wordmask), 1) # [N*M]
        sentencenum = tf.reduce_sum(tf.to_int32(self.sentencemask), 1) # [N]

        docs_emb = tf.reshape(docs_emb, [N*M, JX, dw])
        
        with tf.variable_scope("wordlstm"):
            document, _ = rnn(config.rnn_type, docs_emb, length, d, 
                                scope='document', 
                                dropout_keep_prob=config.input_keep_prob, 
                                wd=config.wd, 
                                is_train=self.is_train)  # [N*M, JX, 2*d]
        if config.debug:
            self.tensor_dict['document'] = document
                

        with tf.variable_scope("pooling_layer"):
            logits = linear(document, 1, True, scope='logits', squeeze=True, 
                            input_keep_prob=config.input_keep_prob, 
                            wd=config.wd, is_train=self.is_train) # [N*M, JX]
            
            document = softsel(document, logits, wordmask, scope='document')
            document = tf.reshape(document, [N, M, 2*d])

        with tf.variable_scope('sentencelstm'):
            document, _ = rnn(config.rnn_type, document, sentencenum, d, 
                                scope='document', 
                                dropout_keep_prob=config.input_keep_prob, 
                                wd=config.wd, 
                                is_train=self.is_train) # [N, M, 2d]

        with tf.variable_scope("pooling_layer2"):
            logits = linear(document, 1, True, scope='logits', squeeze=True, 
                            input_keep_prob=config.input_keep_prob, 
                            wd=config.wd, is_train=self.is_train) # [N, M]

            document = softsel(document, logits, self.sentencemask, 'document') # [N, d]
        
        with tf.variable_scope("hidden_layer"):
            d = document.get_shape().as_list()[-1]
            document = F(document, d, activation=tf.nn.tanh, scope='document', 
                        input_keep_prob=config.input_keep_prob, 
                        wd=config.wd, is_train=self.is_train)
        
        with tf.variable_scope("logits_layer"):
            logits = linear(document, config.n_classes, True, 
                            wd=config.wd, scope='logits') # [N, n_classes]
            
        self.logits = logits
        

    def build_objective(self):
        config = self.config
        labels = tf.one_hot(self.label, self.config.n_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        tf.add_to_collection("losses", self.loss)
        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')

    def build_metrics(self):
        self.correct = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int32(tf.argmax(self.logits, axis=1)), self.label)))
        self.err = tf.abs(tf.to_int32(tf.argmax(self.logits, axis=1)) - self.label)
        self.mse = tf.reduce_sum(self.err * self.err)
        self.err = tf.reduce_sum(self.err)
        
    def build_optimizer(self):
        config = self.config
        self.optimizer = get_optimizer(config, self.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads_and_vars = self.clip_gradient(config, grads_and_vars)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        if config.reinforced:
            
            reinforced_train_op = 

            self.reinforced_train_op = reinforced_train_op


    def clip_gradient(self, config, grads_and_vars):
        if config.grad_clipping is not None:
            grads, vars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, config.grad_clipping)
            grads_and_vars = zip(grads, vars)
        return grads_and_vars

    def init_variable(self, sess):
        sess.run(tf.global_variables_initializer())

    def get_feed_dict(self, config, batch, learning_rate=0.001, mode='train'):
        feed_dict = {}

        if mode == 'train':
            feed_dict[self.is_train] = np.array(True)
            feed_dict[self.learning_rate] = np.array(learning_rate)
        else:
            feed_dict[self.is_train] = np.array(False)

        feed_dict[self.docs] = batch[0]
        feed_dict[self.label] = batch[1]
        feed_dict[self.wordmask] = batch[2] 
        feed_dict[self.sentencemask] = batch[3]

        if config.use_user_info and config.use_product_info:
            feed_dict[self.usr] = batch[4]
            feed_dict[self.prd] = batch[5]

        return feed_dict

    def train(self, sess, trainset, devset, testset):
        # import ipdb; ipdb.set_trace()
        config = self.config
        best_dev_acc = 0.0
        for epoch in range(config.num_epoch):
            
            train_batches = trainset.gen_mini_batch(config.batch_size)
            
            for batch in train_batches:
                global_step = sess.run(self.global_step) + 1
                feed_dict = self.get_feed_dict(config, batch, learning_rate=config.learning_rate)

                if config.debug:
                    fetch_ops = [self.tensor_dict['grad']]
                    fetches = sess.run(fetch_ops, feed_dict=feed_dict)

                if epoch < 100:
                    fetch_ops = [self.loss, self.train_op]
                else:
                    fetch_ops = [self.loss, self.reinforced_train_op]
                    
                loss, _ = sess.run(fetch_ops, feed_dict=feed_dict)
                if global_step % config.print_iter == 0:
                    self.logger.info("{}/{}: loss = {:.4f}".format(epoch, global_step, loss))
                
                if global_step % config.eval_iter == 0:
                    
                    self.logger.info("+" * 50)
                    train_acc, train_mae, train_rmse = self.evaluate(sess, trainset, num_batches=30)
                    self.logger.info("<<<<<<<<======== train set ========>>>>>>>>>")
                    self.logger.info('Accuracy = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}'.format(train_acc, train_mae, train_rmse))

                    dev_acc, dev_mae, dev_rmse = self.evaluate(sess, devset)
                    self.logger.info("<<<<<<<<======== dev set ========>>>>>>>>>")
                    self.logger.info('Accuracy = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}'.format(dev_acc, dev_mae, dev_rmse))

                    test_acc, test_mae, test_rmse = self.evaluate(sess, testset)
                    self.logger.info("<<<<<<<<======== test set ========>>>>>>>>>")
                    self.logger.info('Accuracy = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}'.format(test_acc, test_mae, test_rmse))
                    
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        self.save(sess)
                        self.logger.info("*" * 40)
                        self.logger.info("Saving model at step {}.".format(global_step))
                        self.logger.info("best dev_acc = {:.4f}, test_acc = {:.4f}".format(best_dev_acc, test_acc))
                        self.logger.info("*" * 40)
                    self.logger.info("+" * 50)
    
    def evaluate(self, sess, devset, num_batches=None):
        config = self.config
        n_batches = 0
        dev_batches = devset.gen_mini_batch(config.batch_size // 4) 
        tot, correct, mae, rmse= 0.0, 0.0, 0.0, 0.0
        for batch in dev_batches:
            feed_dict = self.get_feed_dict(config, batch, mode='test')
            cor, mis, mse = sess.run([self.correct, self.err, self.mse], feed_dict=feed_dict)
            correct += cor
            mae += mis
            rmse += mse
            tot += batch[1].shape[0]
            
            n_batches += 1
            if num_batches is not None and n_batches > num_batches:
                break

        dev_acc = float(correct) / float(tot)
        dev_mae = float(mae) / float(tot)
        dev_rmse = math.sqrt(float(rmse) / float(tot))
        return dev_acc, dev_mae, dev_rmse

    
    def restore(self, sess):
        config = self.config
        save_path = "../model/{}/{}".format(config.dataname, config.model_name)
        if config.load_step > 0:
            save_path = "../model/{}/{}-{}".format(config.dataname, config.model_name, config.load_step)
        else:
            checkpoint = tf.train.get_checkpoint_state("../model/{}/".format(config.dataname))
            assert checkpoint is not None, "cannot load checkpoint at ../model/{}/".format(config.dataname)
            save_path = checkpoint.model_checkpoint_path
        self.saver.restore(sess, save_path)

    def save(self, sess):
        config = self.config
        if tf.gfile.Exists('../model/{}'.format(config.dataname)):
            save_path = "../model/{}/{}".format(config.dataname, config.model_name)
        else:
            tf.gfile.MkDir('../model/{}/{}'.format(config.dataname, config.dataname))
        self.saver.save(sess, save_path, self.global_step)