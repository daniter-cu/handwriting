import numpy as np
import theano
import logging as log

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/raccoon")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import argparse
import cPickle
import time


from data import char2int
from utilities import plot_generated_sequences_for_endpoint

floatX = theano.config.floatX

# TODO : Don't hard code this
model_pkl_path = os.path.dirname(os.path.realpath(__file__))+'/cpu_test/f_sampling.pkl'
BIAS = 0.5

class HWGenerator(object):
    def __init__(self):
        self.model, self.f_sampling, self.dict_char2int = \
            cPickle.load(open(model_pkl_path, 'r'))

    def generate(self, text):
        log.debug("Rending text "+ text)
        sample_string = []
        sample_string.append(str(text.strip())+' ')
        cond, cond_mask = char2int(sample_string, self.dict_char2int)

        n_samples = len(sample_string)
        pt_ini_mat = np.zeros((n_samples, 3), floatX)
        h_ini_mat = np.zeros((n_samples, self.model.n_hidden), floatX)
        k_ini_mat = np.zeros((n_samples, self.model.n_mixt_attention), floatX)
        w_ini_mat = np.zeros((n_samples, self.model.n_chars), floatX)

        log.debug('Generating... ')
        beg = time.time()
        pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen = self.f_sampling(
            pt_ini_mat, cond, cond_mask,
            h_ini_mat, k_ini_mat, w_ini_mat, BIAS)
        log.debug('done in {} seconds'.format(time.time()-beg))
        p_gen = np.swapaxes(p_gen, 1, 2)
        mats = [(a_gen, 'alpha'), (k_gen, 'kapa'), (p_gen, 'phi'),
                (w_gen, 'omega')]
        log.debug( 'Printing...')
        beg = time.clock()
        img = plot_generated_sequences_for_endpoint(
                pt_gen, mats,
                mask_gen)

        log.debug( 'done in {} seconds'.format(time.time() - beg) )
        return img
