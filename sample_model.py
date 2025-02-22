import numpy as np
import theano

import sys
sys.path.append("raccoon")

import argparse
import cPickle
import time

from data import char2int
from utilities import plot_generated_sequences, plot_generated_sequences_from_file

floatX = theano.config.floatX


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Sample handriting model')
    parser.add_argument('-f', '--f_sampling_path')
    parser.add_argument('-s', '--sample_string', default='Hello world')
    parser.add_argument('-b', '--bias', type=float, default=0.5)
    parser.add_argument('-n', '--n_times', type=int, default=4)
    parser.add_argument('-l', '--sample_file', default=None)

    options = parser.parse_args()
    sample_file = options.sample_file
    if sample_file:
        print sample_file

    print 'Unpickling... ',
    model, f_sampling, dict_char2int = \
        cPickle.load(open(options.f_sampling_path, 'r'))
    print 'done.'

    n = options.n_times
    sample_string = [options.sample_string + ' '] * n
    if sample_file:
        sample_string = []
        with open(sample_file, "r") as f:
            for line in f:
                sample_string.append(str(line.strip())+' ')

    cond, cond_mask = char2int(sample_string, dict_char2int)

    n_samples = len(sample_string)
    pt_ini_mat = np.zeros((n_samples, 3), floatX)
    h_ini_mat = np.zeros((n_samples, model.n_hidden), floatX)
    k_ini_mat = np.zeros((n_samples, model.n_mixt_attention), floatX)
    w_ini_mat = np.zeros((n_samples, model.n_chars), floatX)

    print 'Generating... ',
    beg = time.time()
    pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen = f_sampling(
        pt_ini_mat, cond, cond_mask,
        h_ini_mat, k_ini_mat, w_ini_mat, options.bias)
    print 'done in {} seconds'.format(time.time()-beg)
    p_gen = np.swapaxes(p_gen, 1, 2)
    mats = [(a_gen, 'alpha'), (k_gen, 'kapa'), (p_gen, 'phi'),
            (w_gen, 'omega')]
    print 'Printing...',
    beg = time.clock()
    if not sample_file:
        plot_generated_sequences(
                pt_gen, mats,
                mask_gen, folder_path='./',
                file_name='test_outputs/a_just_now')
    else:
        plot_generated_sequences_from_file(
                pt_gen, mats,
                mask_gen, folder_path='./test_outputs/')

    print 'done in {} seconds'.format(time.time() - beg)
