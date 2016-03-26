import os

import theano
import theano.tensor as T
import numpy as np

from raccoon import Extension

from data import char2int
from utilities import plot_batch

floatX = theano.config.floatX

class Sampler(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 fun_pred, n_hidden, apply_at_the_start=True,
                 apply_at_the_end=True, n_samples=4):
        super(Sampler, self).__init__(name_extension, freq,
                                      apply_at_the_end=apply_at_the_end,
                                      apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name
        self.fun_pred = fun_pred
        self.n_samples = n_samples
        self.n_hidden = n_hidden

    def execute_virtual(self, batch_id):
        sample = self.fun_pred(np.zeros((self.n_samples, 3), floatX),
                               np.zeros((self.n_samples, self.n_hidden), floatX))

        plot_batch(sample,
                   folder_path=self.folder_path,
                   file_name='{}_'.format(batch_id) + self.file_name)

        return ['executed']


class SamplerCond(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 model, sample_strings, dict_char2int, bias, bias_value=0.5,
                 apply_at_the_start=True, apply_at_the_end=True):
        super(SamplerCond, self).__init__(name_extension, freq,
                                          apply_at_the_end=apply_at_the_end,
                                          apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name

        self.sample_strings = sample_strings
        n_samples = len(sample_strings)
        self.dict_char2int = dict_char2int

        self.bias = bias
        self.bias_value = bias_value

        # Symbolic variables
        seq_str = T.matrix('str_input', 'int32')
        seq_str_mask = T.matrix('str_mask', floatX)
        coord_ini = T.matrix('coord_pred', floatX)
        h_ini_pred, w_ini_pred, k_ini_pred = model.create_sym_init_states()

        # Debug test values
        f_s_str = 7
        n_hidden, dim_char = model.n_hidden, model.dim_char
        n_mixt_attention = model.n_mixt_attention
        coord_ini.tag.test_value = np.zeros((n_samples, 3), floatX)
        h_ini_pred.tag.test_value = np.zeros((n_samples, n_hidden), floatX)
        w_ini_pred.tag.test_value = np.zeros((n_samples, dim_char), floatX)
        k_ini_pred.tag.test_value = np.zeros((n_samples, n_mixt_attention), floatX)
        seq_str.tag.test_value = np.zeros((f_s_str, n_samples), dtype='int32')
        seq_str_mask.tag.test_value = np.ones((f_s_str, n_samples), dtype=floatX)

        # Create graph
        coord_gen, w_gen, updates_pred = model.prediction(
                coord_ini, seq_str, seq_str_mask,
                h_ini_pred, w_ini_pred, k_ini_pred)

        # Compile function
        self.f_sampling = theano.function([coord_ini, seq_str, seq_str_mask,
                                           h_ini_pred, w_ini_pred, k_ini_pred],
                                          [coord_gen, w_gen],
                                          updates=updates_pred)

        # Initial values
        self.coord_ini_mat = np.zeros((n_samples, 3), floatX)
        self.h_ini_mat = np.zeros((n_samples, n_hidden), floatX)
        self.w_ini_mat = np.zeros((n_samples, dim_char), floatX)
        self.k_ini_mat = np.zeros((n_samples, n_mixt_attention), floatX)

    def execute_virtual(self, batch_id):

        bias_val_pre = self.bias.get_value()
        self.bias.set_value(self.bias_value)

        cond, cond_mask = char2int(self.sample_strings, self.dict_char2int)

        sample, w_gen = self.f_sampling(
                self.coord_ini_mat, cond, cond_mask,
                self.h_ini_mat, self.w_ini_mat, self.k_ini_mat)

        plot_batch(sample,
                   folder_path=self.folder_path,
                   file_name='{}_'.format(batch_id) + self.file_name)

        self.bias.set_value(bias_val_pre)

        return ['executed']
