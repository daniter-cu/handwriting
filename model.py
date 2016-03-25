import theano
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from lasagne.init import GlorotNormal, Orthogonal

from raccoon.archi import GRULayer, PositionAttentionLayer

from data import char2int

theano.config.floatX = 'float32'
floatX = theano.config.floatX
np.random.seed(42)


def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


class MixtureGaussians2D:
    def __init__(self, n_in, n_mixtures, initializer, eps=1e-5):
        self.n_mixtures = n_mixtures
        self.eps = eps

        n_out = (n_mixtures +  # proportions
                 n_mixtures * 2 +  # means
                 n_mixtures * 2 +  # stds
                 n_mixtures +  # correlations
                 1)  # bernoulli

        self.w = shared(initializer.sample((n_in, n_out)), 'w_mixt')
        self.b = shared(np.random.normal(
                0, 0.001, size=(n_out, )).astype(floatX), 'b_mixt')

        self.bias = shared(np.float32(.0))

        self.params = [self.w, self.b]

    def compute_parameters(self, h):
        """
        h: (batch or batch*seq, features)
        """
        n = self.n_mixtures
        out = T.dot(h, self.w) + self.b
        prop = T.nnet.softmax(out[:, :n]*(1 + self.bias))
        mean_x = out[:, n:n*2]
        mean_y = out[:, n*2:n*3]
        std_x = T.exp(out[:, n*3:n*4] - self.bias) + self.eps
        std_y = T.exp(out[:, n*4:n*5] - self.bias) + self.eps
        rho = T.tanh(out[:, n*5:n*6])
        rho = (1+rho + self.eps) / (2 + 2*self.eps) - 1
        bernoulli = T.nnet.sigmoid(out[:, -1])
        bernoulli = (bernoulli + self.eps) / (1 + 2*self.eps)

        return prop, mean_x, mean_y, std_x, std_y, rho, bernoulli

    def prediction(self, h):
        srng = RandomStreams(seed=42)

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h)

        mode = T.argmax(srng.multinomial(pvals=prop, dtype=prop.dtype), axis=1)

        v = T.arange(0, mean_x.shape[0])
        m_x = mean_x[v, mode]
        m_y = mean_y[v, mode]
        s_x = std_x[v, mode]
        s_y = std_y[v, mode]
        r = rho[v, mode]
        # cov = r * (s_x * s_y)

        normal = srng.normal((h.shape[0], 2))
        x = normal[:, 0]
        y = normal[:, 1]

        # x_n = T.shape_padright(s_x * x + cov * y + m_x)
        # y_n = T.shape_padright(s_y * y + cov * x + m_y)

        x_n = T.shape_padright(m_x + s_x * x)
        y_n = T.shape_padright(m_y + s_y * (x * r + y * T.sqrt(1.-r**2)))

        uniform = srng.uniform((h.shape[0],))
        pin = T.shape_padright(T.cast(bernoulli > uniform, floatX))

        return T.concatenate([x_n, y_n, pin], axis=1)

    def apply(self, h_seq, mask_seq, tg_seq):
        """
        h_seq: (seq, batch, features)
        mask_seq: (seq, batch)
        tg_seq: (seq, batch, features=3)
        """
        h_seq = T.reshape(h_seq, (-1, h_seq.shape[-1]))
        tg_seq = T.reshape(tg_seq, (-1, tg_seq.shape[-1]))
        mask_seq = T.reshape(mask_seq, (-1,))

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h_seq)

        tg_x = T.addbroadcast(tg_seq[:, 0:1], 1)
        tg_y = T.addbroadcast(tg_seq[:, 1:2], 1)
        tg_pin = tg_seq[:, 2]

        tg_x_s = (tg_x - mean_x) / std_x
        tg_y_s = (tg_y - mean_y) / std_y

        z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s
        buff = 1-rho**2

        tmp = (-z / (2 * buff) -
               T.log(2*np.pi) - T.log(std_x) - T.log(std_y) - 0.5*T.log(buff) +
               T.log(prop))

        c = (-logsumexp(tmp, axis=1) -
             tg_pin * T.log(bernoulli) -
             (1-tg_pin) * T.log(1 - bernoulli))

        c = T.sum(c * mask_seq) / T.sum(mask_seq)

        max_prop = T.argmax(prop, axis=1).mean()
        max_prop.name = 'max_prop'

        std_max_prop = T.argmax(prop, axis=1).std()
        std_max_prop.name = 'std_max_prop'

        return c, [max_prop, std_max_prop]


class UnconditionedModel:
    def __init__(self, gain_ini, n_hidden, n_mixtures):
        ini = GlorotNormal(gain_ini)
        # ini = Orthogonal(gain_ini)

        self.gru_layer = GRULayer(3, n_hidden, ini)

        self.mixture = MixtureGaussians2D(n_hidden, n_mixtures, ini)

        self.params = self.gru_layer.params + self.mixture.params

    def apply(self, seq_coord, seq_mask, seq_tg, h_ini):

        seq_h, scan_updates = self.gru_layer.apply(seq_coord, seq_mask, h_ini)
        loss, monitoring = self.mixture.apply(seq_h, seq_mask, seq_tg)

        h_mean = seq_h.mean()
        h_mean.name = 'h_mean'
        monitoring.append(h_mean)

        return loss, [(h_ini, seq_h[-1])] + scan_updates, monitoring

    def prediction(self, coord_ini, h_ini, n_steps=500):

        def gru_step(coord_pre, h_pre):

            h = self.gru_layer.step(coord_pre, h_pre,
                                    mask=None, process_inputs=True)

            coord = self.mixture.prediction(h)

            return coord, h

        res, scan_updates = theano.scan(
                fn=gru_step,
                outputs_info=[coord_ini, h_ini],
                n_steps=n_steps)

        return res[0], scan_updates


class ConditionedModel:
    def __init__(self, gain_ini, n_hidden, dim_char, n_mixt_attention,
                 n_mixtures):
        """
        Parameters
        ----------
        n_mixt_attention: int
            Number of mixtures used
        """
        self.n_hidden = n_hidden
        self.dim_char = dim_char
        self.n_mixt_attention = n_mixt_attention
        self.n_mixtures = n_mixtures

        ini = GlorotNormal(gain_ini)
        # ini = Orthogonal(gain_ini)

        self.pos_layer = PositionAttentionLayer(
                GRULayer(3+self.dim_char, n_hidden, ini),
                self.dim_char,
                self.n_mixt_attention, ini)

        self.gru_layer = GRULayer(n_hidden+self.dim_char, n_hidden, ini)

        self.mixture = MixtureGaussians2D(n_hidden,
                                          n_mixtures, ini)

        self.params = self.pos_layer.params + self.mixture.params

    def create_shared_init_states(self, batch_size):

        def create_shared(size, name):
            return theano.shared(np.zeros(size, floatX), name)

        h_ini = create_shared((batch_size, self.n_hidden), 'h_ini')
        w_ini = create_shared((batch_size, self.dim_char), 'w_ini')
        k_ini = create_shared((batch_size, self.n_mixt_attention), 'k_ini')
        h2_ini = create_shared((batch_size, self.n_hidden), 'h2_ini')

        return h_ini, w_ini, k_ini, h2_ini

    def reset_shared_init_states(self, h_ini, w_ini, k_ini, h2_ini, batch_size):

        def set_value(var, size):
            var.set_value(np.zeros(size, dtype=floatX))

        set_value(h_ini, (batch_size, self.n_hidden))
        set_value(w_ini, (batch_size, self.dim_char))
        set_value(k_ini, (batch_size, self.n_mixt_attention))
        set_value(h2_ini, (batch_size, self.n_hidden))

    def create_sym_init_states(self):
        h_ini_pred = T.matrix('h_ini_pred', floatX)
        w_ini_pred = T.matrix('w_ini_pred', floatX)
        k_ini_pred = T.matrix('k_ini_pred', floatX)
        h2_ini_pred = T.matrix('h2_ini_pred', floatX)
        return h_ini_pred, w_ini_pred, k_ini_pred, h2_ini_pred

    def apply(self, seq_coord, seq_mask, seq_tg, seq_str, seq_str_mask,
              h_ini, w_ini, k_ini, h2_ini):

        # seq_str will have shape (seq_length, batch_size, dim_char]
        seq_str = T.eye(self.dim_char, dtype=floatX)[seq_str]

        (seq_h, seq_w, seq_k), scan_updates = self.pos_layer.apply(
                seq_coord, seq_mask, seq_str, seq_str_mask,
                h_ini, w_ini, k_ini)

        seq_h_conc = T.concatenate([seq_h, seq_w], axis=-1)

        seq_h2, scan_updates2 = self.gru_layer.apply(
                seq_h_conc, seq_mask, h2_ini)

        loss, monitoring = self.mixture.apply(seq_h2, seq_mask, seq_tg)

        updates = [(h_ini, seq_h[-1]), (w_ini, seq_w[-1]), (k_ini, seq_k[-1]),
                   (h2_ini, seq_h2[-1])]

        seq_h_mean = seq_h.mean()
        seq_h_mean.name = 'seq_h_mean'

        seq_w_mean = seq_w.mean()
        seq_w_mean.name = 'seq_w_mean'

        argmax_seq_w = T.argmax(seq_w, axis=-1)
        argmax_seq_w_mean = argmax_seq_w.mean()
        argmax_seq_w_mean.name = 'argmax_seq_w_mean'
        argmax_seq_w_std = argmax_seq_w.std()
        argmax_seq_w_std.name = 'argmax_seq_w_std'

        max_seq_w = T.max(seq_w, axis=-1)
        max_seq_w_mean = max_seq_w.mean()
        max_seq_w_mean.name = 'max_seq_w_mean'
        max_seq_w_std = max_seq_w.std()
        max_seq_w_std.name = 'max_seq_w_std'

        seq_k_mean = seq_k.mean()
        seq_k_mean.name = 'seq_k_mean'

        seq_k_std = seq_k.std()
        seq_k_std.name = 'seq_k_std'

        monitoring.extend([seq_h_mean, seq_w_mean, seq_k_mean, seq_k_std,
                           argmax_seq_w_mean, argmax_seq_w_std,
                           max_seq_w_mean, max_seq_w_std])

        return loss, updates + scan_updates + scan_updates2, monitoring

    def prediction(self, coord_ini, seq_str, seq_str_mask,
                   h_ini, w_ini, k_ini, h2_ini, n_steps=500):

        seq_str = T.eye(self.dim_char, dtype=floatX)[seq_str]

        def scan_step(coord_pre, h_pre, w_pre, k_pre, h2_pre,
                      seq_str, seq_str_mask):

            h, w, k = self.pos_layer.step(coord_pre, h_pre, w_pre, k_pre,
                                          seq_str, seq_str_mask, mask=None)

            seq_h_conc = T.concatenate([h, w], axis=-1)

            h2 = self.gru_layer.step(seq_h_conc, h2_pre, mask=None,
                                     process_inputs=True)

            coord = self.mixture.prediction(h2)

            return coord, h, w, k, h2

        res, scan_updates = theano.scan(
                fn=scan_step,
                outputs_info=[coord_ini, h_ini, w_ini, k_ini, h2_ini],
                non_sequences=[seq_str, seq_str_mask],
                n_steps=n_steps)

        return res[0], scan_updates

