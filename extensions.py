import os

import theano
import numpy as np

from raccoon import Extension

from utilities import plot_batch

floatX = theano.config.floatX

class Sampler(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 fun_pred, h_ini, apply_at_the_start=True,
                 apply_at_the_end=True, n_samples=4):
        super(Sampler, self).__init__(name_extension, freq,
                                      apply_at_the_end=apply_at_the_end,
                                      apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name
        self.fun_pred = fun_pred
        self.h_ini = h_ini
        self.n_samples = n_samples

    def execute_virtual(self, batch_id):
        previous_value = self.h_ini.get_value()
        self.h_ini.set_value(np.zeros_like(previous_value, floatX))
        sample = self.fun_pred(np.zeros((previous_value.shape[0], 3), floatX))
        self.h_ini.set_value(previous_value)

        # print 'mean x: {}'.format(sample[:, :, 0].mean())
        # print 'std x: {}'.format(sample[:, :, 0].std())
        # print 'mean y: {}'.format(sample[:, :, 1].mean())
        # print 'std y: {}'.format(sample[:, :, 1].std())

        plot_batch(sample[:, 0:self.n_samples],
                   folder_path=self.folder_path,
                   file_name='{}_'.format(batch_id) + self.file_name)

        return ['executed']
