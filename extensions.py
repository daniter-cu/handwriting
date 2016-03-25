import os

import theano
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
                 fun_pred, sample_strings,
                 coord_ini_mat, h_ini_mat, w_ini_mat, k_ini_mat,
                 dict_char2int, bias, bias_value=0.5, apply_at_the_start=True,
                 apply_at_the_end=True):
        super(SamplerCond, self).__init__(name_extension, freq,
                                          apply_at_the_end=apply_at_the_end,
                                          apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name
        self.fun_pred = fun_pred
        self.sample_strings = sample_strings
        self.dict_char2int = dict_char2int
        self.coord_ini_mat = coord_ini_mat
        self.h_ini_mat = h_ini_mat
        self.w_ini_mat = w_ini_mat
        self.k_ini_mat = k_ini_mat
        self.bias = bias
        self.bias_value = bias_value

    def execute_virtual(self, batch_id):

        bias_val_pre = self.bias.get_value()
        self.bias.set_value(self.bias_value)

        cond, cond_mask = char2int(self.sample_strings, self.dict_char2int)

        sample = self.fun_pred(
                self.coord_ini_mat, cond, cond_mask,
                self.h_ini_mat, self.w_ini_mat, self.k_ini_mat)

        plot_batch(sample,
                   folder_path=self.folder_path,
                   file_name='{}_'.format(batch_id) + self.file_name)

        self.bias.set_value(bias_val_pre)

        return ['executed']
