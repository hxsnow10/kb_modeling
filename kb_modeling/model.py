#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from collections import defaultdict
from data_utils import TripleDataset
from config import Config

class TransEModel(object):

    def __init__(self, config):
        entity_total=config.entity_total
        relation_total=config.relation_total
        batch_size = 100
        size = config.hidden_size
        margin = config.margin

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
            self.predict = pos
        
        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
            
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.init=tf.initialize_all_variables()
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        self.saver = tf.train.Saver()
        
class TransRModel(object):

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        sizeE = config.hidden_sizeE
        sizeR = config.hidden_sizeR
        margin = config.margin

        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [batch_size])
            self.pos_t = tf.placeholder(tf.int32, [batch_size])
            self.pos_r = tf.placeholder(tf.int32, [batch_size])
            self.neg_h = tf.placeholder(tf.int32, [batch_size])
            self.neg_t = tf.placeholder(tf.int32, [batch_size])
            self.neg_r = tf.placeholder(tf.int32, [batch_size])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, sizeE], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_matrix = tf.get_variable(name = "rel_matrix", shape = [relation_total, sizeE * sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, sizeE, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, sizeE, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, sizeR])
            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, sizeE, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, sizeE, 1])
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, sizeR])          
            matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r), [-1, sizeR, sizeE])

            pos_h_e = tf.reshape(tf.batch_matmul(matrix, pos_h_e), [-1, sizeR])
            pos_t_e = tf.reshape(tf.batch_matmul(matrix, pos_t_e), [-1, sizeR])
            neg_h_e = tf.reshape(tf.batch_matmul(matrix, neg_h_e), [-1, sizeR])
            neg_t_e = tf.reshape(tf.batch_matmul(matrix, neg_t_e), [-1, sizeR])

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

class TransDModel(object):

    def calc(self, e, t, r):
        return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
            pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
            pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)

            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
            neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
            neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
            neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)

            pos_h_e = self.calc(pos_h_e, pos_h_t, pos_r_t)
            pos_t_e = self.calc(pos_t_e, pos_t_t, pos_r_t)
            neg_h_e = self.calc(neg_h_e, neg_h_t, neg_r_t)
            neg_t_e = self.calc(neg_t_e, neg_t_t, neg_r_t)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

