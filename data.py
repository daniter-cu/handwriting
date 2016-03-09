import os
import h5py
import numpy as np
import theano


theano.config.floatX = 'float32'
floatX = theano.config.floatX


def load_data(filename='hand_training.hdf5'):
    data_folder = os.path.join(os.environ['DATA_PATH'], 'handwriting')
    training_data_file = os.path.join(data_folder, filename)
    train_data = h5py.File(training_data_file, 'r')

    coord_seq = train_data['pt_seq'][:]
    coord_idx = train_data['pt_idx'][:]
    strings_seq = train_data['str_seq'][:]
    strings_idx = train_data['str_idx'][:]

    return coord_seq, coord_idx, strings_seq, strings_idx


def create_generator(shuffle, batch_size, seq_coord, coord_idx,
                     seq_strings, strings_idx):
    n_seq = coord_idx.shape[0]
    idx = np.arange(n_seq)
    np.random.shuffle(idx)

    coord_idx = coord_idx[idx]
    strings_idx = strings_idx[idx]

    def generator():
        for i in range(0, n_seq-batch_size, batch_size):
            yield create_batch(slice(i, i+batch_size),
                               seq_coord, coord_idx, seq_strings, strings_idx)

    return generator



def create_batch(slice, coord, coord_idx, strings, strings_idx, M=None):
    """
    the slice represents the minibatch
    - coord: shape (number points, 3)
    - coord_idx: shape (number of sequences, 2): the starting and end points of
        each sequence
    """
    if not M:
        M = 10000

    # Two big sequenes
    pt_idxs = coord_idx[slice]
    str_idxs = strings_idx[slice]

    longuest_pt_seq = max([b - a for a, b in pt_idxs])
    longuest_pt_seq = min(M, longuest_pt_seq)
    pt_batch = np.zeros((longuest_pt_seq, len(pt_idxs), 3), dtype=floatX)
    pt_mask_batch = np.zeros((longuest_pt_seq, len(pt_idxs)), dtype=floatX)

    str_batch = []

    for i, (pt_seq, str_seq) in enumerate(zip(pt_idxs, str_idxs)):
        pts = np.array(coord[pt_seq[0]:pt_seq[1]])
        limit2 = min(pts.shape[0], longuest_pt_seq)
        pt_batch[:limit2, i] = pts[:limit2]
        pt_mask_batch[:limit2, i] = 1

        strs = strings[str_seq[0]:str_seq[1]]
        str_batch.append(strs)

    str_batch = np.array(str_batch)

    return pt_batch, pt_mask_batch, str_batch
