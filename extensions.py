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
                 fun_pred, coord_ini, h_ini,
                 apply_at_the_end=True):
        super(Sampler, self).__init__(name_extension, freq,
                                    apply_at_the_end=apply_at_the_end)
        self.folder_path = folder_path
        self.file_name = file_name
        self.fun_pred = fun_pred
        self.coord_ini = coord_ini
        self.h_ini = h_ini

    def execute_virtual(self, batch_id):
        previous_value = self.h_ini.get_value()
        self.h_ini.set_value(np.zeros_like(previous_value, floatX))
        sample = self.fun_pred(self.coord_ini)
        self.h_ini.set_value(previous_value)

        plot_batch(sample[:, 0:1],
                   folder_path=self.folder_path,
                   file_name='{}_'.format(batch_id) + self.file_name)

        # file_path = os.path.join(
        #     self.folder_path, self.file_name + '.png')

        return ['executed']
