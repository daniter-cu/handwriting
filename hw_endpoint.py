import numpy
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
from utilities import plot_generated_sequences, plot_generated_sequences_from_file

floatX = theano.config.floatX

# TODO : Don't hard code this
model_pkl_path = os.path.dirname(os.path.realpath(__file__))+'/cpu_test/f_sampling.pkl'

class HWGenerator(object):
    def __init__(self):
        self.model, self.f_sampling, self.dict_char2int = \
            cPickle.load(open(model_pkl_path, 'r'))

    def generate(self, text):
        log.debug("Rending text "+ text)
