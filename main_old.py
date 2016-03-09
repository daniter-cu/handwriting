import os
import sys
import numpy as np
import h5py

import theano
# theano.config.optimizer = 'None'
import theano.tensor as T
from theano import shared
from theano.gradient import grad_clip
from lasagne.init import GlorotNormal

from data_generation import create_batch, create_generator
from utilities import plot_seq

from raccoon import *

theano.config.floatX = 'float32'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.01
n_hidden = 400
n_mixture = 20
gain = 0.1
batch_size = 100

# DATA
data_folder = os.path.join(os.environ['DATA_PATH'], 'handwriting')
training_data_file = os.path.join(data_folder, 'hand_training.hdf5')
train_data = h5py.File(training_data_file, 'r')

tr_coord_seq = train_data['points'][:]
tr_coord_idx = train_data['points_seq'][:]
tr_strings_seq = train_data['strings'][:]
tr_strings_idx = train_data['strings_seq'][:]


# M_x = 8.16648
# M_y = 0.111465
# s_x = 41.9263
# s_y = 37.0967
# pt_batch, pt_mask_batch, str_batch = \
#     create_batch(slice(1000, 1100), tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx)
#
# pt_batch[:, :, 0] = s_x*pt_batch[:, :, 0] + M_x
# pt_batch[:, :, 1] = s_y*pt_batch[:, :, 1] + M_y
#
# for i in range(100):
#     plot_seq(pt_batch[:, i], pt_mask_batch[:, i].astype(bool))


# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_tg = T.tensor3('tg', floatX)
seq_mask = T.matrix('tg', floatX)
h = T.matrix('hidden_state', floatX)

n_inputs = 3

ini = GlorotNormal(gain)




class GRULayer():
    def __init__(self, n_in, n_h, n_out):
        W_r = shared(ini.sample((n_inputs, n_hidden)))
        U_r = shared(ini.sample((n_inputs, n_hidden)))
        b_r = shared(np.zeros((n_hidden,), floatX))



# shared variables
w_in = shared(ini.sample((n_inputs, n_hidden)))
w_rec = shared(ini.sample((n_hidden, n_hidden)))
b_rec = shared(np.zeros((n_hidden,), floatX))

n_out = (n_mixture +  # proportions
         n_mixture * 2 +  # means
         n_mixture * 2 +  # stds
         n_mixture +  # correlations
         1)  # bernoulli
# proportions of each mixture
w_out = shared(ini.sample((n_hidden, n_out)))
b_out = shared(np.zeros((n_out,), floatX))

params = w_in, w_rec, b_rec, w_out, b_out


def step(coords, tg, mask, h_pre, w_in, w_rec, b_rec, w_out, b_out):
    h_new = T.tanh(T.dot(coords, w_in) + T.dot(h_pre, w_rec) + b_rec)
    h = grad_clip(mask[:, None]*h_new + (1-mask[:, None])*h_pre, -10, 10)

    out = T.dot(h, w_out) + b_out
    prop = T.nnet.softmax(out[:, :n_mixture])
    mean_x = out[:, n_mixture: n_mixture*2]
    mean_y = out[:, n_mixture*2: n_mixture*3]
    std_x = T.exp(out[:, n_mixture*3: n_mixture*4])
    std_y = T.exp(out[:, n_mixture*4: n_mixture*5])
    rho = T.tanh(out[:, n_mixture*5: n_mixture*6])
    bernoulli = T.nnet.sigmoid(out[:, -1])

    tg_x = tg[:, 0:1]
    tg_x = T.addbroadcast(tg_x, 1)
    tg_y = tg[:, 1:2]
    tg_y = T.addbroadcast(tg_y, 1)
    tg_pin = tg[:, 2]

    tg_x_s = (tg_x - mean_x) / std_x
    tg_y_s = (tg_y - mean_y) / std_y

    z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s

    buff = 1-rho**2

    p = T.exp(-z / (2 * buff)) / (2*np.pi*std_x*std_y*T.sqrt(buff))

    c = (-T.log(T.sum(p * prop, axis=1)) -
         tg_pin * T.log(bernoulli) - (1-tg_pin) * T.log(1 - bernoulli))

    return h, c.mean()


seq_out, updates_scan = theano.scan(
        fn=step,
        sequences=[seq_coord, seq_tg, seq_mask],
        outputs_info=[h, None],
        non_sequences=params,  # les poids utilises
        strict=True)
hidden_states = seq_out[0]
losses = seq_out[1]

loss = losses.sum()
loss.name = 'negll'

hidden_states2 = (hidden_states**2).mean()
hidden_states2.name = 'hid'

grads = T.grad(loss, params)

updates_params = []
for p, g in zip(params, grads):
    updates_params.append((p, p - learning_rate * g))
updates_all = updates_scan + updates_params


train_monitor = TrainMonitor(1, [seq_coord, seq_tg, seq_mask, h], [loss, hidden_states2],
                             updates_all)

train_m = Trainer(train_monitor, [], [])

batch_gen = create_generator(True, batch_size,
                       tr_coord_seq, tr_coord_idx,
                       tr_strings_seq, tr_strings_idx)

it = 0
sys.stdout.flush()
epoch = 0
try:
    while True:
        epoch += 1
        for pt_batch, pt_mask_batch, str_batch in batch_gen():
            if it == 1000:
                sys.exit()
            print pt_batch.max()
            print pt_batch.min()
            pt_batch[pt_batch>4] = 4
            pt_batch[pt_batch<-4] = -4
            h_mat = np.zeros((batch_size, n_hidden), dtype=floatX)
            res = train_m.process_batch(epoch, it,
                                        pt_batch[1:], pt_batch[:-1],
                                        pt_mask_batch[1:], h_mat)
            it += 1
            # print params[1].get_value().sum()
            if res:
                train_m.finish(it)
                sys.exit()
except KeyboardInterrupt:
    print 'Training interrupted by user.'
    train_m.finish(it)





# seq_coord_mat = np.random.normal(0, 1, (seq_size, batch_size, 3))
# seq_coord_mat[:, :, -1] = 0
# seq_tg_mat = seq_coord_mat[1:]
# seq_coord_mat = seq_coord_mat[:-1]
# h_mat = np.random.normal(0, 1, (batch_size, n_hidden))
#
# res = f(seq_coord_mat, seq_tg_mat, h_mat)
# print res



# print 'compiling...'
# f = theano.function([seq_coord, seq_tg, seq_mask, h], loss,
#                     updates=updates_all)
# print 'done!'


# h_mat = np.random.normal(0, init, (batch_size, n_hidden))
# pt_batch, pt_mask_batch, str_batch = create_batch(
#         slice(0, batch_size), tr_coord_seq, tr_coord_idx, tr_strings_seq,
#         tr_strings_idx)
# res = f(pt_batch[1:], pt_batch[:-1], pt_mask_batch[1:], h_mat)
# print res