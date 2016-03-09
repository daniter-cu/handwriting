from theano import shared
from theano.gradient import grad_clip
from lasagne.init import GlorotNormal

from data import create_generator, load_data

from raccoon import *

theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
floatX = theano.config.floatX
np.random.seed(42)

# CONFIG
learning_rate = 0.1
n_hidden = 400
n_mixture = 20
gain = 0.1
batch_size = 100

# DATA
tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx = \
    load_data('hand_training.hdf5')


# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_tg = T.tensor3('tg', floatX)
seq_mask = T.matrix('tg', floatX)
h = T.matrix('hidden_state', floatX)

n_inputs = 3

ini = GlorotNormal(gain)


class GRULayer():
    def __init__(self, n_in, n_h):
        self.w = shared(ini.sample((n_inputs, n_hidden)), 'w')
        self.u = shared(ini.sample((n_hidden, n_hidden)), 'u')
        self.b = shared(np.zeros((n_hidden,), floatX), 'b')

        self.wr = shared(ini.sample((n_inputs, n_hidden)), 'wr')
        self.ur = shared(ini.sample((n_hidden, n_hidden)), 'ur')
        self.br = shared(np.zeros((n_hidden,), floatX), 'br')

        self.wu = shared(ini.sample((n_inputs, n_hidden)), 'wu')
        self.uu = shared(ini.sample((n_hidden, n_hidden)), 'uu')
        self.bu = shared(np.zeros((n_hidden,), floatX), 'bu')

        self.params = [self.w, self.u, self.b,
                       self.wr, self.ur, self.br,
                       self.wu, self.uu, self.bu]

    @staticmethod
    def apply(x, h, w, u, b, wr, ur, br, wu, uu, bu):
        r_gate = T.nnet.sigmoid(T.dot(x, wr) + T.dot(h, ur) + br)
        u_gate = T.nnet.sigmoid(T.dot(x, wu) + T.dot(r_gate * h, uu) + bu)

        h_new = T.tanh(T.dot(x, w) + T.dot(r_gate * h, u) + b)

        return (1-u_gate)*h + u_gate*h_new

# shared variables
gru_layer = GRULayer(n_inputs, n_hidden)

n_out = (n_mixture +  # proportions
         n_mixture * 2 +  # means
         n_mixture * 2 +  # stds
         n_mixture +  # correlations
         1)  # bernoulli
# proportions of each mixture
w_out = shared(ini.sample((n_hidden, n_out)))
b_out = shared(np.zeros((n_out,), floatX))

params = gru_layer.params + [w_out, b_out]


def step(coords, tg, mask, h_pre,
         w, u, b, wr, ur, br, wu, uu, bu,
         w_out, b_out):

    h_new = gru_layer.apply(coords, h_pre, w, u, b, wr, ur, br, wu, uu, bu)
    h = mask[:, None]*h_new + (1-mask[:, None])*h_pre
    h = grad_clip(h, -10, 10)

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

    eps = 1e-8

    tg_x_s = (tg_x - mean_x) / (std_x + eps)
    tg_y_s = (tg_y - mean_y) / (std_y + eps)

    z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s

    buff = 1-rho**2 + eps

    p = T.exp(-z / (2 * buff)) / (2*np.pi*std_x*std_y*T.sqrt(buff) + eps)

    c = (-T.log(T.sum(p * prop, axis=1)+eps) -
         tg_pin * T.log(bernoulli) - (1-tg_pin) * T.log(1 - bernoulli))

    c = c[mask > 0]

    return h, c.mean()



# def prediction(coords, mask, h_pre,
#                W, U, b, W_r, U_r, b_r, W_u, U_u, b_u,
#                w_out, b_out):
#
#     h_new = gru_layer.apply(coords, h_pre, W, U, b, W_r, U_r, b_r, W_u, U_u, b_u)
#     h = mask[:, None]*h_new + (1-mask[:, None])*h_pre
#     h = grad_clip(h, -10, 10)
#
#     out = T.dot(h, w_out) + b_out
#     prop = T.nnet.softmax(out[:, :n_mixture])
#     mean_x = out[:, n_mixture: n_mixture*2]
#     mean_y = out[:, n_mixture*2: n_mixture*3]
#     std_x = T.exp(out[:, n_mixture*3: n_mixture*4])
#     std_y = T.exp(out[:, n_mixture*4: n_mixture*5])
#     rho = T.tanh(out[:, n_mixture*5: n_mixture*6])
#     bernoulli = T.nnet.sigmoid(out[:, -1])
#
#     eps = 1e-8
#
#     tg_x_s = (tg_x - mean_x) / (std_x + eps)
#     tg_y_s = (tg_y - mean_y) / (std_y + eps)
#
#     z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s
#
#     buff = 1-rho**2 + eps
#
#     p = T.exp(-z / (2 * buff)) / (2*np.pi*std_x*std_y*T.sqrt(buff) + eps)
#
#     c = (-T.log(T.sum(p * prop, axis=1)+eps) -
#          tg_pin * T.log(bernoulli) - (1-tg_pin) * T.log(1 - bernoulli))
#
#     return h, c.mean()




seq_out, updates_scan = theano.scan(
        fn=step,
        sequences=[seq_coord, seq_tg, seq_mask],
        outputs_info=[h, None],
        non_sequences=params,  # les poids utilises
        strict=True)
hidden_states = seq_out[0]
losses = seq_out[1]

loss = losses.mean()
loss.name = 'negll'

hidden_states2 = (hidden_states**2).mean()
hidden_states2.name = 'hid'

grads = T.grad(loss, params)

updates_params = []
for p, g in zip(params, grads):
    updates_params.append((p, p - learning_rate * g))
updates_all = updates_scan + updates_params


train_monitor = TrainMonitor(10, [seq_coord, seq_tg, seq_mask, h], [loss, hidden_states2],
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
            # if it == 20:
            #     sys.exit()
            h_mat = np.zeros((batch_size, n_hidden), dtype=floatX)
            res = train_m.process_batch(epoch, it,
                                        pt_batch[1:], pt_batch[:-1],
                                        pt_mask_batch[1:], h_mat)
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