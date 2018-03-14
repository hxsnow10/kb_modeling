#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from collections import defaultdict
from data_utils import TripleDataset
from config import config
import math

class BaseModel(object):

    def __init__(self):
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.build_network()
        self.build_loss()
        self.build_others()

    def build_others(self):
        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.start_learning_rate, global_step,
            config.decay_steps, config.decay_rate, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.step_summaries = tf.summary.merge_all()
        # optimizer = tf.train.AdamOptimizer(0.01)
        # self.train_op=optimizer.minimize(self.loss)
        self.init=tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
class MarginBasedModel(BaseModel):
    
    def build_loss(self): 
        if config.L1_flag:
            pos = tf.reduce_sum(abs( self.pos_dist ), 1, keep_dims = True)
            neg = tf.reduce_sum(abs( self.neg_dist ), 1, keep_dims = True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((self.pos_dist) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((self.neg_dist) ** 2, 1, keep_dims = True)
            self.predict = pos
        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + config.margin, 0))
        
class TransEModel(MarginBasedModel):

    def build_network(self):
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [config.entity_total, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [config.relation_total, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            def calc(h, t, r):
                return tf.nn.embedding_lookup(self.ent_embeddings, h),\
                    tf.nn.embedding_lookup(self.ent_embeddings, t),\
                    tf.nn.embedding_lookup(self.rel_embeddings, r)
            
            self.pos_h_e, self.pos_t_e, self.pos_r_e = calc(self.pos_h, self.pos_t, self.pos_r)
            self.neg_h_e, self.neg_t_e, self.neg_r_e = calc(self.neg_h, self.neg_t, self.neg_r)
        self.pos_dist = self.pos_h_e + self.pos_r_e - self.pos_t_e
        self.neg_dist = self.neg_h_e + self.neg_r_e - self.neg_t_e

class TransRModel(MarginBasedModel):

    def build_network(self):
        sizeE = config.hidden_size
        sizeR = config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [config.entity_total, sizeE], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [config.relation_total, sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_matrix = tf.get_variable(name = "rel_matrix", shape = [config.relation_total, sizeE * sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            
            def calc(h,t,r):
                
                h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, h), [-1, sizeE, 1])
                t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, t), [-1, sizeE, 1])
                r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, r), [-1, sizeR])
                matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r), [-1, sizeR, sizeE])
                h_e = tf.reshape(tf.matmul(matrix, h_e), [-1, sizeR])
                t_e = tf.reshape(tf.matmul(matrix, t_e), [-1, sizeR])
                return h_e, t_e, r_e
            self.pos_h_e, self.pos_t_e, self.pos_r_e = calc(self.pos_h, self.pos_t, self.pos_r)
            self.neg_h_e, self.neg_t_e, self.neg_r_e = calc(self.neg_h, self.neg_t, self.neg_r)
        self.pos_dist = self.pos_h_e + self.pos_r_e - self.pos_t_e
        self.neg_dist = self.neg_h_e + self.neg_r_e - self.neg_t_e

class TransDModel(MarginBasedModel):

    def calc(self, e, t, r):
        return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

    def build_network(self):
        size=config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [config.entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [config.relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [config.entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [config.relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            def calc(h,t,r):
                h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
                t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)
                r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
                h_t = tf.nn.embedding_lookup(self.ent_transfer, h)
                t_t = tf.nn.embedding_lookup(self.ent_transfer, t)
                r_t = tf.nn.embedding_lookup(self.rel_transfer, r)
                h_e = self.calc(h_e, h_t, r_t)
                t_e = self.calc(t_e, t_t, r_t)
                return h_e, t_e, r_e
            self.pos_h_e, self.pos_t_e, self.pos_r_e = calc(self.pos_h, self.pos_t, self.pos_r)
            self.neg_h_e, self.neg_t_e, self.neg_r_e = calc(self.neg_h, self.neg_t, self.neg_r)
        self.pos_dist = self.pos_h_e + self.pos_r_e - self.pos_t_e
        self.neg_dist = self.neg_h_e + self.neg_r_e - self.neg_t_e


class DistMul(MarginBasedModel):

    def build_network(self):
        size = config.hidden_size
        sizeE = config.hidden_size
        sizeR = config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [config.entity_total, sizeE],
             initializer = tf.random_uniform_initializer(-math.pow(1.0/size, 0.33), math.pow(1.0/size, 0.33)))
            # , initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [config.relation_total, sizeE],
             initializer = tf.random_uniform_initializer(-math.pow(1.0/size, 0.33), math.pow(1.0/size, 0.33)))
            # , initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            def calc(h,t,r):
                h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, h), [-1, 1, sizeE])
                t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, t), [-1, sizeE,1])
                r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, r), [-1, sizeE])
                r_e = tf.matrix_diag(r_e)
                score=tf.matmul(tf.matmul(h_e,r_e), t_e)
                return score

            self.pos_score=tf.reshape(calc(self.pos_h, self.pos_t, self.pos_r),[-1,1])
            self.neg_score=tf.reshape(calc(self.neg_h, self.neg_t, self.neg_r),[-1,1])

    def build_loss(self):
        self.predict=self.pos_score
        with tf.name_scope("output"):
            self.losses = self.pos_score - self.neg_score
            self.loss = tf.reduce_sum(tf.maximum(self.pos_score - self.neg_score+1,0))
            # self.loss = tf.reduce_sum(self.pos_score - self.neg_score)
            ''' 
            self.logits=tf.concat([self.pos_score,self.neg_score],1)
            self.labels=tf.concat([tf.ones([config.batch_size,1]), tf.zeros([config.batch_size,1])],1)
            self.loss=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss=tf.reduce_sum(self.loss)
            '''
            # self.loss = tf.reduce_sum(-(self.pos_score-self.neg_score)/(abs(self.pos_score)+abs(self.neg_score)))
