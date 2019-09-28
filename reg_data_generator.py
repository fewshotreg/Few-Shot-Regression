"""
Code for generating and loading data
"""

import numpy as np
import os
import random
import tensorflow as tf
import pickle
from scipy import signal
# from utils import point_source
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


class SinusoidGenerator(object):
    def __init__(self, train=True, bmaml=False):
        self.few_k_shot = FLAGS.few_k_shot
        self.bmaml = bmaml
        if train:
            f = open('sinusoidal_train.pkl', 'rb')
        else:
            f = open('sinusoidal_test.pkl', 'rb')
        import pickle
        data = pickle.load(f)
        f.close()
        self.data = dict()
        self.data['train_x'] = data['x'][:, :self.few_k_shot, :]
        self.data['train_y'] = data['y'][:, :self.few_k_shot, :]
        self.data['test_x'] = data['x'][:, 20:, :]
        self.data['test_y'] = data['y'][:, 20:, :]

        print('load data: train_x', self.data['train_x'].shape, 'test_x', self.data['test_x'].shape, 'train_y',
              self.data['train_y'].shape, 'test_y', self.data['test_y'].shape)

        self.dim_input = 1
        self.dim_output = 1

    def generate_batch(self, indx):
        context_x = self.data['train_x'][indx]
        context_y = self.data['train_y'][indx]
        target_x = self.data['test_x'][indx]
        target_y = self.data['test_y'][indx]
        if self.bmaml:
            leader_x = np.concatenate([context_x, target_x], 1)
            leader_y = np.concatenate([context_y, target_y], 1)
            return context_x, context_y, leader_x, leader_y, target_x, target_y
        return context_x, context_y, target_x, target_y


class Mnist():
    def __init__(self, num_shot, num_val, train=True, bmaml=False):
        self.num_shot = num_shot
        self.bmaml = bmaml
        self.dim_input = 2
        self.dim_output = 1
        import pickle
        f = open('mnist.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        if train:
            data_str = 'train'
        else:
            data_str = 'test'
        self.data = dict()
        self.data['train_x'] = data['meta_' + data_str + '_x'][:, :num_shot, :]
        self.data['train_y'] = data['meta_' + data_str + '_y'][:, :num_shot, :]
        self.data['test_x'] = data['meta_' + data_str + '_x'][:, num_shot:num_shot+num_val, :]
        self.data['test_y'] = data['meta_' + data_str + '_y'][:, num_shot:num_shot+num_val, :]

        print('load data: train_x', self.data['train_x'].shape, 'test_x', self.data['test_x'].shape, 'train_y',
              self.data['train_y'].shape, 'test_y', self.data['test_y'].shape)

    def generate_batch(self, indx):
        context_x = self.data['train_x'][indx]
        context_y = self.data['train_y'][indx]
        target_x = self.data['test_x'][indx]
        target_y = self.data['test_y'][indx]
        if self.bmaml:
            leader_x = np.concatenate([context_x, target_x], 1)
            leader_y = np.concatenate([context_y, target_y], 1)
            return context_x, context_y, leader_x, leader_y, target_x, target_y
        return context_x, context_y, target_x, target_y


class MnistNP():
    def __init__(self, num_shot, num_val, train=True):
        self.num_shot = num_shot
        import pickle
        f = open('mnist.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        if train:
            data_str = 'train'
        else:
            data_str = 'test'
        self.data = dict()
        self.data['train_x'] = data['meta_' + data_str + '_x'][:, :num_shot, :]
        self.data['train_y'] = data['meta_' + data_str + '_y'][:, :num_shot, :]
        self.data['test_x'] = data['meta_' + data_str + '_x'][:, num_shot:num_shot+num_val, :]
        self.data['test_y'] = data['meta_' + data_str + '_y'][:, num_shot:num_shot+num_val, :]

        print('load data: train_x', self.data['train_x'].shape, 'test_x', self.data['test_x'].shape, 'train_y',
              self.data['train_y'].shape, 'test_y', self.data['test_y'].shape)

    def generate_batch(self, indx):
        context_x = self.data['train_x'][indx]
        context_y = self.data['train_y'][indx]
        target_x = self.data['test_x'][indx]
        target_y = self.data['test_y'][indx]
        return context_x, context_y, target_x, target_y

