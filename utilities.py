import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


M_x = 8.18586
M_y = 0.11457
s_x = 40.3719
s_y = 37.0466


def plot_batch(pt_batch, pt_mask_batch=None, use_mask=False, show=False, folder_path=None, file_name=None):
    batch_normed = np.zeros_like(pt_batch, dtype=pt_batch.dtype)
    batch_normed[:] = pt_batch
    batch_normed[:, :, 0] = s_x * batch_normed[:, :, 0] + M_x
    batch_normed[:, :, 1] = s_y * batch_normed[:, :, 1] + M_y

    print 'mean x: {}'.format(batch_normed[:, :, 0].mean())
    print 'std x: {}'.format(batch_normed[:, :, 0].std())
    print 'mean y: {}'.format(batch_normed[:, :, 1].mean())
    print 'std y: {}'.format(batch_normed[:, :, 1].std())

    n_samples = batch_normed.shape[1]
    fig = plt.figure(figsize=(n_samples*3, 10))

    for i in range(n_samples):
        mask_term = np.array([])
        if use_mask:
            mask_term = pt_mask_batch[:, i].astype(bool)
        subplot = fig.add_subplot(n_samples, 1, i + 1)
        plot_seq(subplot, batch_normed[:, i], mask_term)

    if folder_path and file_name:
        fig.savefig(os.path.join(folder_path, file_name + '.png'))
    if show:
        plt.show()
    plt.close()


def plot_seq(subplot, seq_pt, seq_mask=np.array([]), norm=False):
    if norm:
        seq_normed = np.zeros_like(seq_pt, dtype=seq_pt.dtype)
        seq_normed[:] = seq_pt
        seq_normed[:, 0] = s_x * seq_normed[:, 0] + M_x
        seq_normed[:, 1] = s_y * seq_normed[:, 1] + M_y
    else:
        seq_normed = seq_pt

    coord = seq_normed[:, 0:2]
    penup = seq_normed[:, 2].astype(bool)

    pos = np.where(penup)[0]+1

    if seq_mask.size:
        coord = np.cumsum(coord[seq_mask.astype(bool)], axis=0)
    else:
        coord = np.cumsum(coord, axis=0)
    coord = np.insert(coord, pos, [np.nan, np.nan], axis=0)

    subplot.plot(coord[:, 0], -coord[:, 1])
