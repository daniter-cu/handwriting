import theano
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from lasagne.init import GlorotNormal, Orthogonal

from raccoon.archi import GRULayer

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
        self.b = shared(np.zeros((n_out,), floatX), 'b_mixt')

        self.params = [self.w, self.b]

    def compute_parameters(self, h, w, b):
        """
        h: (batch or batch*seq, features)
        """
        n = self.n_mixtures
        out = T.dot(h, w) + b
        prop = T.nnet.softmax(out[:, :n])
        mean_x = out[:, n:n*2]
        mean_y = out[:, n*2:n*3]
        std_x = T.exp(out[:, n*3:n*4]) + self.eps
        std_y = T.exp(out[:, n*4:n*5]) + self.eps
        rho = T.tanh(out[:, n*5:n*6])
        rho = (1+rho + self.eps) / (2 + 2*self.eps) - 1
        bernoulli = T.nnet.sigmoid(out[:, -1])
        bernoulli = (bernoulli + self.eps) / (1 + 2*self.eps)

        return prop, mean_x, mean_y, std_x, std_y, rho, bernoulli

    def prediction(self, h, w, b):
        srng = RandomStreams(seed=42)

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h, w, b)

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
        mask_seq = T.reshape(mask_seq, (-1, mask_seq.shape[-1]))

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h_seq, self.w, self.b)

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

        c = c[mask_seq > 0]

        max_prop = T.argmax(prop, axis=1).mean()
        max_prop.name = 'max_prop'

        std_max_prop = T.argmax(prop, axis=1).std()
        std_max_prop.name = 'std_max_prop'

        s_x_m = std_x.mean()
        s_x_m.name = 'std_x'

        s_y_m = std_y.mean()
        s_y_m.name = 'std_y'

        m_x_m = mean_x.mean()
        m_x_m.name = 'mean_x'

        m_y_m = mean_y.mean()
        m_y_m.name = 'mean_y'

        return c.mean(), [m_x_m, m_y_m, s_x_m, s_y_m, max_prop, std_max_prop]


class Model1:
    def __init__(self, gain_ini, n_hidden, n_mixtures):
        ini = GlorotNormal(gain_ini)
        # ini = Orthogonal(gain_ini)

        self.gru_layer = GRULayer(3, n_hidden, ini)

        self.mixture = MixtureGaussians2D(n_hidden, n_mixtures, ini)

        self.params = self.gru_layer.params + self.mixture.params

        self.scan_updates = []

    def apply(self, seq_coord, seq_mask, seq_tg, h_ini):

        seq_h, scan_updates = self.gru_layer.apply(seq_coord, seq_mask, h_ini)
        loss, monitoring = self.mixture.apply(seq_h, seq_mask, seq_tg)

        h_mean = seq_h.mean()
        h_mean.name = 'h_mean'
        monitoring.append(h_mean)

        return loss, [(h_ini, seq_h[-1])] + scan_updates, monitoring

    def prediction(self, coord_ini, h_ini, n_steps=500):

        def gru_step(coord_pre, h_pre,
                     w, wr, wu, b, br, bu, u, ur, uu, w_mixt, b_mixt):

            x_in = T.dot(coord_pre, w) + b
            x_r = T.dot(coord_pre, wr) + br
            x_u = T.dot(coord_pre, wu) + bu

            r_gate = T.nnet.sigmoid(x_r + T.dot(h_pre, ur))
            u_gate = T.nnet.sigmoid(x_u + T.dot(h_pre, uu))

            h_new = T.tanh(x_in + T.dot(r_gate * h_pre, u))

            h = (1-u_gate)*h_pre + u_gate*h_new

            coord = self.mixture.prediction(h, w_mixt, b_mixt)

            return coord, h

        res, scan_updates = theano.scan(
                fn=gru_step,
                outputs_info=[coord_ini, h_ini],
                non_sequences=self.params,  # les poids utilises
                n_steps=n_steps,
                strict=True)

        return res[0], scan_updates

