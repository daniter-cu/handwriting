import theano
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from lasagne.init import GlorotNormal

from raccoon.archi import GRULayer

theano.config.floatX = 'float32'
floatX = theano.config.floatX
np.random.seed(42)


class MixtureGaussians2D:
    def __init__(self, n_in, n_mixtures, initializer):
        self.n_mixtures = n_mixtures

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
        mean_x = out[:, n: n*2]
        mean_y = out[:, n*2: n*3]
        std_x = T.exp(out[:, n*3: n*4])
        std_y = T.exp(out[:, n*4: n*5])
        rho = T.tanh(out[:, n*5: n*6])
        bernoulli = T.nnet.sigmoid(out[:, -1])

        return prop, mean_x, mean_y, std_x, std_y, rho, bernoulli

    def prediction(self, h, w, b):
        srng = RandomStreams(seed=42)
        normal = srng.normal((h.shape[0], 2))
        uniform = srng.uniform((h.shape[0],))

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h, w, b)

        mode = T.argmax(prop, axis=1)

        v = T.arange(0, mean_x.shape[0])
        m_x = mean_x[v, mode]
        m_y = mean_y[v, mode]
        s_x = std_x[v, mode]
        s_y = std_y[v, mode]
        r = rho[v, mode]
        cov = r * (s_x * s_y)

        x = normal[:, 0]
        y = normal[:, 1]

        x_n = T.shape_padright(s_x * x + cov * y + m_x)
        y_n = T.shape_padright(s_y * y + cov * x + m_y)

        pin = T.shape_padright(T.cast(bernoulli > uniform, floatX))

        return T.concatenate([x_n, y_n, pin], axis=1)

    def apply(self, h_seq, mask_seq, tg_seq, eps=1e-5):
        """
        h_seq: (seq, batch, features)
        mask_seq: (seq, batch)
        tg_seq: (seq, batch, features=3)
        """
        h_seq = T.reshape(h_seq, (-1, h_seq.shape[2]))
        tg_seq = T.reshape(tg_seq, (-1, tg_seq.shape[2]))
        mask_seq = T.reshape(mask_seq, (-1, mask_seq.shape[1]))

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h_seq, self.w, self.b)

        tg_x = tg_seq[:, 0:1]
        tg_x = T.addbroadcast(tg_x, 1)
        tg_y = tg_seq[:, 1:2]
        tg_y = T.addbroadcast(tg_y, 1)
        tg_pin = tg_seq[:, 2]

        tg_x_s = (tg_x - mean_x) / (std_x + eps)
        tg_y_s = (tg_y - mean_y) / (std_y + eps)

        z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s

        buff = 1-rho**2

        p = T.exp(-z / (2 * buff + eps)) / (2*np.pi*std_x*std_y*T.sqrt(buff) + eps)

        c = (-T.log(T.sum(p * prop, axis=1) + eps) -
             tg_pin * T.log(bernoulli + eps) -
             (1-tg_pin) * T.log(1 - bernoulli + eps))

        c = c[mask_seq > 0]

        return c.mean()


class Model1:
    def __init__(self, gain_ini, n_hidden, n_mixtures):
        ini = GlorotNormal(gain_ini)

        self.gru_layer = GRULayer(3, n_hidden, ini)

        self.mixture = MixtureGaussians2D(n_hidden, n_mixtures, ini)

        self.params = self.gru_layer.params + self.mixture.params

        self.scan_updates = []

    def apply(self, seq_coord, seq_mask, seq_tg, h_ini):

        seq_h, scan_updates = self.gru_layer.apply(seq_coord, seq_mask, h_ini)
        loss = self.mixture.apply(seq_h, seq_mask, seq_tg)

        return loss, [(h_ini, seq_h[-1])] + scan_updates

    def prediction(self, coord_ini, h_ini, n_steps=500):

        gru = self.gru_layer

        def gru_step(coord_pre, h_pre,
                     w, wr, wu, u, b, ur, br, uu, bu, w_mixt, b_mixt):

            x_in = T.dot(coord_pre, gru.w)
            x_r = T.dot(coord_pre, gru.wr)
            x_u = T.dot(coord_pre, gru.wu)

            r_gate = T.nnet.sigmoid(x_r + T.dot(h_pre, ur) + br)
            u_gate = T.nnet.sigmoid(x_u + T.dot(r_gate * h_pre, uu) + bu)

            h_new = T.tanh(x_in + T.dot(r_gate * h_pre, u) + b)

            h = (1-u_gate)*h_pre + u_gate*h_new

            coord = self.mixture.prediction(h, w_mixt, b_mixt)

            return coord, h

        res, scan_updates = theano.scan(
                fn=gru_step,
                outputs_info=[coord_ini, h_ini],
                non_sequences=self.params,  # les poids utilises
                n_steps=n_steps,
                strict=True)

        return res[0], [(h_ini, res[1][-1])] + scan_updates

