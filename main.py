import getpass

from lasagne.updates import adam

from raccoon import *

from data import create_generator, load_data, extract_sequence
from model import Model1
from extensions import Sampler

from matplotlib import pyplot as plt
from utilities import plot_seq, plot_batch

theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.05
n_hidden = 900
n_mixtures = 20
gain = 0.1
batch_size = 20
chunk = 20
every = 10000
tmp_path = os.environ.get('TMP_PATH')
dump_path = os.path.join(tmp_path, 'handwriting',
                         str(np.random.randint(0, 100000000, 1)[0]))
if not os.path.exists(dump_path):
    os.makedirs(dump_path)

# DATA
tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx = \
    load_data('hand_training.hdf5')


# pt_batch, pt_mask_batch, str_batch = \
#     extract_sequence(slice(0, 4),
#                    tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx)
# plot_batch(pt_batch, pt_mask_batch, use_mask=True, show=True)


# MODEL CREATION
# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_tg = T.tensor3('tg', floatX)
seq_mask = T.matrix('mask', floatX)
h_ini = theano.shared(np.zeros((batch_size, n_hidden), floatX), 'hidden_state')
# h_ini = T.matrix('hidden_state', floatX)

model = Model1(gain, n_hidden, n_mixtures)
loss, updates, monitoring = model.apply(seq_coord, seq_mask, seq_tg, h_ini)
loss.name = 'negll'

params = model.params
grads = T.grad(loss, params)


# # Clip norm of the gradients
# def clip_norm(g, c, n):
#     if c > 0:
#         g = T.switch(n >= c, g * c / n, g)
#     return g
# norm = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
# grads = [clip_norm(g, 1, norm) for g in grads]


def gradient_clipping(grads, rescale=5.):
    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    scaling_num = rescale
    scaling_den = tensor.maximum(rescale, grad_norm)
    scaling = scaling_num / scaling_den
    return [g * scaling for g in grads]

grads = gradient_clipping(grads)

# updates_params = adam(grads, params, 0.0003)

updates_params = []
for p, g in zip(params, grads):
    updates_params.append((p, p - learning_rate * g))

updates_all = updates + updates_params

coord_ini = T.matrix('coord', floatX)
pred, updates_pred = model.prediction(coord_ini, h_ini)
f_sampling = theano.function([coord_ini], pred, updates=updates_pred)

# MONITORING
train_monitor = TrainMonitor(every, [seq_coord, seq_tg, seq_mask],
                             [loss] + monitoring, updates_all)

sampler = Sampler('sampler', every, dump_path, 'essai',
                  f_sampling, h_ini)

train_m = Trainer(train_monitor, [sampler], [])

batch_gen = create_generator(
        True, batch_size,
        tr_coord_seq, tr_coord_idx,
        tr_strings_seq, tr_strings_idx, chunk=chunk)

it = 0
epoch = 0
h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
try:
    while True:
        epoch += 1
        for (pt_in, pt_tg, pt_mask, str), next_seq in batch_gen():
            # plt.figure(figsize=(3, 10))
            # plot_seq(plt, pt_in[:, 0], pt_mask[:, 0], True)
            # plt.show()
            res = train_m.process_batch(epoch, it,
                                        pt_in, pt_tg,
                                        pt_mask)
            if next_seq:
                h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
            it += 1
            if res:
                train_m.finish(it)
                sys.exit()
except KeyboardInterrupt:
    print 'Training interrupted by user.'
    train_m.finish(it)
