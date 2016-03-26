import os
import sys
import cPickle

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor
from raccoon.archi.utils import clip_norm_gradients

from data import create_generator, load_data, extract_sequence
from model import ConditionedModel
from extensions import SamplerCond

theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.1
n_hidden = 600
n_chars = 81
n_mixt_attention = 10
n_gaussian_mixtures = 20
gain = 0.01
batch_size = 50  # batch_size
every = 100
sample_strings = ['Jose is a raccoon !']*4
tmp_path = os.environ.get('TMP_PATH')
dump_path = os.path.join(tmp_path, 'handwriting',
                         str(np.random.randint(0, 100000000, 1)[0]))
if not os.path.exists(dump_path):
    os.makedirs(dump_path)

# DATA
tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx = \
    load_data('hand_training.hdf5')
char_dict, inv_char_dict = cPickle.load(open('char_dict.pkl', 'r'))
batch_gen = create_generator(
        True, batch_size,
        tr_coord_seq, tr_coord_idx,
        tr_strings_seq, tr_strings_idx, chunk=None)

# MODEL CREATION
# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_str = T.matrix('str_input', 'int32')
seq_tg = T.tensor3('tg', floatX)
seq_pt_mask = T.matrix('pt_mask', floatX)
seq_str_mask = T.matrix('str_mask', floatX)

# Debug
f_s_pt = 6
f_s_str = 7
seq_coord.tag.test_value = np.zeros((f_s_pt, batch_size, 3), dtype=floatX)
seq_str.tag.test_value = np.zeros((f_s_str, batch_size), dtype='int32')
seq_tg.tag.test_value = np.ones((f_s_pt, batch_size, 3), dtype=floatX)
seq_pt_mask.tag.test_value = np.ones((f_s_pt, batch_size), dtype=floatX)
seq_str_mask.tag.test_value = np.ones((f_s_str, batch_size), dtype=floatX)
# End Debug

model = ConditionedModel(gain, n_hidden, n_chars, n_mixt_attention,
                         n_gaussian_mixtures)
h_ini, w_ini, k_ini = model.create_shared_init_states(batch_size)
loss, updates, monitoring = model.apply(seq_coord, seq_pt_mask, seq_tg,
                                        seq_str, seq_str_mask,
                                        h_ini, w_ini, k_ini)
loss.name = 'negll'


# GRADIENT AND UPDATES
params = model.params
grads = T.grad(loss, params)
grads = clip_norm_gradients(grads)

updates_params = adam(grads, params, 0.0003)

# updates_params = []
# for p, g in zip(params, grads):
#     updates_params.append((p, p - learning_rate * g))

updates_all = updates + updates_params


# MONITORING
train_monitor = TrainMonitor(every, [seq_coord, seq_tg, seq_pt_mask,
                                     seq_str, seq_str_mask],
                             [loss] + monitoring, updates_all)

sampler = SamplerCond('sampler', every, dump_path, 'essai',
                      model, sample_strings, dict_char2int=char_dict,
                      bias=model.mixture.bias, bias_value=0.5)

train_m = Trainer(train_monitor, [sampler], [])


it = 0
epoch = 0
model.reset_shared_init_states(h_ini, w_ini, k_ini, batch_size)
try:
    while True:
        epoch += 1
        for (pt_in, pt_tg, pt_mask, str, str_mask), next_seq in batch_gen():
            res = train_m.process_batch(epoch, it,
                                        pt_in, pt_tg,
                                        pt_mask, str, str_mask)
            if next_seq:
                model.reset_shared_init_states(h_ini, w_ini, k_ini, batch_size)
            it += 1
            if res:
                train_m.finish(it)
                sys.exit()
except KeyboardInterrupt:
    print 'Training interrupted by user.'
    train_m.finish(it)
