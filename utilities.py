import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


M_x = 8.16648
M_y = 0.111465
s_x = 41.9263
s_y = 37.0967


def plot_batch(pt_batch, pt_mask_batch=None, show=False, folder_path=None, file_name=None):
    pt_batch[:, :, 0] = s_x * pt_batch[:, :, 0] + M_x
    pt_batch[:, :, 1] = s_y * pt_batch[:, :, 1] + M_y

    for i in range(pt_batch.shape[1]):
        mask_term = None
        if pt_mask_batch:
            mask_term = pt_mask_batch[:, i].astype(bool)
        plt.figure()
        plot_seq(pt_batch[:, i], mask_term, show)
        if folder_path and file_name:
            plt.savefig(os.path.join(folder_path, file_name + '_{}.png'.format(i)))
        plt.close()


def plot_seq(seq, mask=None, show=False):
    coord = seq[:, 0:2]
    pin = seq[:, 2].astype(bool)

    pos = np.where(pin)[0]+1

    if mask:
        coord = np.cumsum(coord[mask], axis=0)
    else:
        coord = np.cumsum(coord, axis=0)
    coord = np.insert(coord, pos, [np.nan, np.nan], axis=0)

    plt.plot(coord[:, 0], -coord[:, 1])
    if show:
        plt.show()
