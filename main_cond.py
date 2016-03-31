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
from extensions import SamplerCond, SamplingFunctionSaver, ValMonitorHandwriting
from utilities import create_train_tag_values, create_gen_tag_values

theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
theano.config.compute_test_value = 'raise'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.1
n_hidden = 1200
n_chars = 81
n_mixt_attention = 10
n_gaussian_mixtures = 20
gain = 0.01
batch_size = 50  # batch_size
chunk = None
every = 100
every_val = 1000
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
train_batch_gen = create_generator(
        True, batch_size,
        tr_coord_seq, tr_coord_idx,
        tr_strings_seq, tr_strings_idx, chunk=chunk)
val_coord_seq, val_coord_idx, val_strings_seq, val_strings_idx = \
    load_data('hand_training.hdf5')
valid_batch_gen = create_generator(
    True, batch_size,
    val_coord_seq, val_coord_idx,
    val_strings_seq, val_strings_idx, chunk=chunk)

# MODEL CREATION
# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_str = T.matrix('str_input', 'int32')
seq_tg = T.tensor3('tg', floatX)
seq_pt_mask = T.matrix('pt_mask', floatX)
seq_str_mask = T.matrix('str_mask', floatX)
create_train_tag_values(seq_coord, seq_str, seq_tg, seq_pt_mask,
                        seq_str_mask, batch_size)  # for debug


model = ConditionedModel(gain, n_hidden, n_chars, n_mixt_attention,
                         n_gaussian_mixtures)
h_ini, k_ini, w_ini = model.create_shared_init_states(batch_size)
loss, updates_ini, monitoring = model.apply(seq_coord, seq_pt_mask, seq_tg,
                                            seq_str, seq_str_mask,
                                            h_ini, k_ini, w_ini)
loss.name = 'negll'


# GRADIENT AND UPDATES
params = model.params
grads = T.grad(loss, params)
grads = clip_norm_gradients(grads)

updates_params = adam(grads, params, 0.0003)

# updates_params = []
# for p, g in zip(params, grads):
#     updates_params.append((p, p - learning_rate * g))

updates_all = updates_ini + updates_params


# SAMPLING FUNCTION
coord_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias = \
    model.create_sym_init_states()
create_gen_tag_values(model, coord_ini, h_ini_pred, k_ini_pred, w_ini_pred,
                      bias, seq_str, seq_str_mask)  # for debug

(coord_gen, a_gen, k_gen, p_gen, w_gen, mask_gen), updates_pred = \
    model.prediction(coord_ini, seq_str, seq_str_mask,
                     h_ini_pred, k_ini_pred, w_ini_pred, bias=bias)

f_sampling = theano.function(
    [coord_ini, seq_str, seq_str_mask, h_ini_pred, k_ini_pred, w_ini_pred,
     bias], [coord_gen, a_gen, k_gen, p_gen, w_gen, mask_gen],
    updates=updates_pred)


# MONITORING
train_monitor = TrainMonitor(
    every, [seq_coord, seq_tg, seq_pt_mask, seq_str, seq_str_mask],
    [loss] + monitoring, updates_all)

valid_monitor = ValMonitorHandwriting(
    'Validation', every_val, [seq_coord, seq_tg, seq_pt_mask, seq_str,
                              seq_str_mask], [loss] + monitoring,
    valid_batch_gen, updates_ini, model, h_ini, k_ini, w_ini, batch_size,
    apply_at_the_start=False)

sampler = SamplerCond('sampler', every, dump_path, 'essai',
                      model, f_sampling, sample_strings,
                      dict_char2int=char_dict, bias_value=0.5)

sampling_saver = SamplingFunctionSaver(
    valid_monitor, loss, every_val, dump_path, 'f_sampling', model,
    f_sampling, char_dict, apply_at_the_start=True)

train_m = Trainer(train_monitor, [valid_monitor, sampler, sampling_saver], [])


it = epoch = 0
model.reset_shared_init_states(h_ini, k_ini, w_ini, batch_size)
try:
    while True:
        epoch += 1
        for inputs, next_seq in train_batch_gen():
            res = train_m.process_batch(epoch, it, *inputs)

            if next_seq:
                model.reset_shared_init_states(h_ini, k_ini, w_ini, batch_size)
            it += 1
            if res:
                train_m.finish(it)
                sys.exit()
except KeyboardInterrupt:
    print 'Training interrupted by user.'
    train_m.finish(it)
