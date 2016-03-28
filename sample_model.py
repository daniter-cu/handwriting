import os
import sys
import cPickle

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor
from raccoon.archi.utils import clip_norm_gradients

from data import create_generator, load_data, extract_sequence
from model import ConditionedModel
from extensions import SamplerCond, SamplingFunctionSaver
from utilities import create_train_tag_values, create_gen_tag_values


import argparse
import cPickle
import time

from data import char2int
from utilities import plot_batch

floatX = theano.config.floatX


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Sample handriting model')
    parser.add_argument('-f', '--f_sampling_path')
    parser.add_argument('-s', '--sample_string', default='Hello world')
    parser.add_argument('-b', '--bias', type=float, default=0.5)
    parser.add_argument('-n', '--n_times', type=int, default=4)

    options = parser.parse_args()

    print 'Unpickling... ',
    model, f_sampling, dict_char2int = \
        cPickle.load(open(options.f_sampling_path, 'r'))
    print 'done.'

    n = options.n_times
    sample_string = [options.sample_string + ' ']

    cond, cond_mask = char2int(sample_string, dict_char2int)

    n_samples = len(sample_string)
    coord_ini_mat = np.zeros((n_samples, 3), floatX)
    h_ini_mat = np.zeros((n_samples, model.n_hidden), floatX)
    w_ini_mat = np.zeros((n_samples, model.n_chars), floatX)
    k_ini_mat = np.zeros((n_samples, model.n_mixt_attention), floatX)

    print 'Generating... ',
    beg = time.clock()
    sample, w_gen = f_sampling(
        coord_ini_mat, cond, cond_mask,
        h_ini_mat, w_ini_mat, k_ini_mat, options.bias)
    print 'done in {} seconds'.format(time.clock()-beg)
    plot_batch(sample,
               folder_path='./',
               file_name='a_just_now')