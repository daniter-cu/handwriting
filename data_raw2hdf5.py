#!/usr/bin/env python

import time
import os
from os.path import join
import re
from xml.dom.minidom import parse
import numpy as np
import h5py

# Config
dataset = 'validation'

# mean and std used for standardization of the data
M_x = 8.18586
M_y = 0.11457
s_x = 40.3719
s_y = 37.0466

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

old_string = 0
old_point = 0


def read_file(file_path):
    pts = []
    pre_pt = np.array([])
    for trace in parse(file_path).getElementsByTagName('Stroke'):
        for coords in trace.getElementsByTagName('Point'):
            pt = np.array([coords.getAttribute('x').strip(),
                           coords.getAttribute('y').strip(), 0],
                          dtype='float32')
            if not len(pre_pt):
                pre_pt = pt
            diff = pt - pre_pt
            if np.any(diff >= 1000) or np.any(diff <= -1000):
                return False
            diff[-1] = 0
            pre_pt = pt
            pts.append(diff)
        pts[-1][-1] = 1

    return pts, get_target_string(file_path)

last_pt_idx = 0
last_str_idx = 0
for ii, l in enumerate(file(strokes_filename).readlines()):
    if ii == 10:
        break
    file_path = l.strip()
    file_path = join(data_folder, file_path)

    if not len(file_path):
        continue

    res = read_file(file_path)
    if not res:
        continue

    pts, str = res
    pt_seq.extend(pts)
    str_seq.append(str)

    next_pt_idx = last_pt_idx + len(pts)
    pt_idx.append((last_pt_idx, next_pt_idx))
    last_pt_idx = next_pt_idx

    next_str_idx = last_str_idx + len(str)
    str_idx.append((last_str_idx, next_str_idx))
    last_str_idx = next_str_idx

str_seq = "".join(str_seq)
str_seq = str_seq.encode("ascii", "ignore")
pt_seq = np.array(pt_seq, dtype='float32')

print time.clock() - start


# M_x = pt_seq[:, 0].mean()
# M_y = pt_seq[:, 1].mean()
# s_x = pt_seq[:, 0].std()
# s_y = pt_seq[:, 1].std()

# Normalize
pt_seq[:, 0] = (pt_seq[:, 0] - M_x) / s_x
pt_seq[:, 1] = (pt_seq[:, 1] - M_y) / s_y



from utilities import plot_batch
from data import extract_sequence
pt_batch, pt_mask_batch, str_batch = \
    extract_sequence(slice(0, 4), pt_seq, pt_idx, str_seq, str_idx)
plot_batch(pt_batch, pt_mask_batch, use_mask=True, show=True)




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
