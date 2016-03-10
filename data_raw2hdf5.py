#!/usr/bin/env python

import time
import os
from os.path import join
import re
from xml.dom.minidom import parse
import numpy as np
import h5py

# Config
dataset = 'training'

# mean and std used for standardization of the data
M_x = 8.16648
M_y = 0.111465
s_x = 41.9263
s_y = 37.0967

data_folder = join(os.environ['DATA_PATH'], 'handwriting')
strokes_filename = join(data_folder, dataset + '_set.txt')
h5_filename = join(data_folder, 'handwriting_' + dataset + '.hdf5')


def get_target_string(stroke_filename):

    ascii_filename = re.sub('lineStrokes', 'ascii', stroke_filename)
    ascii_filename = re.sub('-[0-9]+\.xml', '.txt', ascii_filename)
    try:
        line_nr = int(re.search('-([0-9]+)\.xml', stroke_filename).group(1))
        lines = [line.strip() for line in open(ascii_filename)]
        return lines[line_nr+lines.index('CSR:') + 1]
    except (AttributeError, IndexError):
        raise SystemExit

start = time.clock()

pt_seq = []  # all sequences concatenated
pt_idx = []  # dim (n_seq, 2) beginning and end of each sequence
str_seq = []  # all sequences concatenated
str_idx = []  # dim (n_seq, 2) beginning and end of each sequence
seqTags = []

old_string = 0
old_point = 0
for ii, l in enumerate(file(strokes_filename).readlines()):
    # if ii == 10:
    #     break
    file_path = l.strip()
    file_path = join(data_folder, file_path)

    if not len(file_path):
        continue

    seqTags.append(file_path)

    seqTxt = get_target_string(file_path)
    str_idx.append((old_string, old_string + len(seqTxt)))
    old_string += len(seqTxt)
    str_seq.append(seqTxt)

    firstCoord = np.array([])
    for trace in parse(file_path).getElementsByTagName('Stroke'):
        for coords in trace.getElementsByTagName('Point'):
            pt = np.array([float(coords.getAttribute('x').strip()), float(coords.getAttribute('y').strip())])
            last = np.array([float(pt[0]), float(pt[1]), 0.0])
            if len(firstCoord) == 0: firstCoord = last
            last = last - firstCoord
            pt_seq.append(last)
        pt_seq[-1][-1] = 1
    pt_idx.append((old_point, len(pt_seq)))
    old_point = len(pt_seq)

str_seq = "".join(str_seq)
str_seq = str_seq.encode("ascii", "ignore")
pt_seq = np.array(pt_seq, dtype='float32')

# Compute offsets instead of absolute positions
firstIx = 0
for i in range(len(pt_idx)):
    l = pt_idx[i][1] - pt_idx[i][0]
    for k in reversed(range(l)):
        pt_seq[firstIx + k][:2] = pt_seq[firstIx + k][:2] - pt_seq[firstIx + k - 1][:2]
    pt_seq[firstIx] = np.array([0, 0, 0])
    firstIx += l

print time.clock() - start


# Visualisation
# from data_generation import create_batch
# from utilities import plot_seq
#
# pt_batch, pt_mask_batch, str_batch = \
#     create_batch(slice(0,100), points, points_seq, strings, strings_seq)
# for i in range(100):
#     plot_seq(pt_batch[:, i], pt_mask_batch[:, i].astype(bool))


idx = np.all(pt_seq <= 100, axis=1)
pt_seq = pt_seq[idx]

# Normalize
pt_seq[:, 0] = (pt_seq[:, 0] - M_x) / s_x
pt_seq[:, 1] = (pt_seq[:, 1] - M_y) / s_y


f = h5py.File(h5_filename, 'w')

ds_points = f.create_dataset(
        "pt_seq",
        (len(pt_seq), 3),
        dtype='float32',
        data=pt_seq)
ds_points_length = f.create_dataset(
        "pt_idx",
        (len(pt_idx), 2),
        dtype='int32',
        data=pt_idx)
ds_strings = f.create_dataset(
        "str_seq",
        (len(str_seq),),
        dtype='S10',
        data=list(str_seq))
ds_strings_length = f.create_dataset(
        "str_idx",
        (len(str_idx), 2),
        dtype='int32',
        data=str_idx)
