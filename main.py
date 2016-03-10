import getpass

from raccoon import *

from data import create_generator, load_data
from model import Model1
from extensions import Sampler

theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.1
n_hidden = 900
n_mixtures = 20
gain = 0.1
batch_size = 100
chunk = 20
tmp_path = os.environ.get('TMP_PATH')
dump_path = os.path.join(tmp_path, 'handwriting',
                         str(np.random.randint(0, 100000000, 1)[0]))
if not os.path.exists(dump_path):
    os.makedirs(dump_path)

# DATA
tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx = \
    load_data('hand_training.hdf5')

# MODEL CREATION
# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_tg = T.tensor3('tg', floatX)
seq_mask = T.matrix('tg', floatX)
h_ini = theano.shared(np.zeros((batch_size, n_hidden), floatX), 'hidden_state')
# h_ini = T.matrix('hidden_state', floatX)

model = Model1(gain, n_hidden, n_mixtures)
loss, updates = model.apply(seq_coord, seq_mask, seq_tg, h_ini)
loss.name = 'negll'

params = model.params
grads = T.grad(loss, params)

updates_params = []
for p, g in zip(params, grads):
    updates_params.append((p, p - learning_rate * g))
updates_all = updates + updates_params

coord_ini = T.matrix('coord', floatX)
pred, updates_pred = model.prediction(coord_ini, h_ini)
f_sampling = theano.function([coord_ini], pred, updates=updates_pred)

# MONITORING
train_monitor = TrainMonitor(10, [seq_coord, seq_tg, seq_mask], [loss],
                             updates_all)

coord_ini_mat = np.zeros((batch_size, 3), floatX)
sampler = Sampler('sampler', 10, dump_path, 'essai',
                  f_sampling, coord_ini_mat, h_ini)

train_m = Trainer(train_monitor, [sampler], [])

batch_gen = create_generator(True, batch_size,
                       tr_coord_seq, tr_coord_idx,
                       tr_strings_seq, tr_strings_idx, chunk=chunk)

it = 0
sys.stdout.flush()
epoch = 0
h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
try:
    while True:
        epoch += 1
        for (pt_batch, pt_mask_batch, str_batch), next_seq in batch_gen():
            # if it == 20:
            #     sys.exit()
            res = train_m.process_batch(epoch, it,
                                        pt_batch[1:], pt_batch[:-1],
                                        pt_mask_batch[1:])
            if next_seq:
                h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
            it += 1
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