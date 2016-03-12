import numpy as np

import theano
import theano.tensor as T
from lasagne.init import GlorotNormal

from model import MixtureGaussians2D

floatX = theano.config.floatX
np.random.seed(42)

learning_rate = 0.01
n_in = 3
n_mixtures = 12
bs = 10

input = T.matrix('input', floatX)
tg = T.matrix('tg', floatX)
mask = T.vector('mask', floatX)

ini = GlorotNormal(0.1)
mixture = MixtureGaussians2D(n_in, n_mixtures, ini)
loss, mon = mixture.apply(input, mask, tg)

grads = T.grad(loss, mixture.params)
updates_params = []
for p, g in zip(mixture.params, grads):
    updates_params.append((p, p - learning_rate * g))

f = theano.function([input, tg, mask], [loss], updates=updates_params)

tg_mat = np.ones((bs, n_in), floatX)
tg_mat[:, 0] = 0.2
tg_mat[:, 1] = -0.1
tg_mat[:, 2] = 0
mask_mat = np.ones((bs,), floatX)


for i in range(10):
    in_mat = np.random.normal(0, 1, (bs, n_in)).astype(floatX)
    in_mat[:, 2] = 0
    print f(in_mat, tg_mat, mask_mat)


prediction = mixture.prediction(input, mixture.w, mixture.b)

f_pred = theano.function([input], prediction)

in_pred_mat = np.random.normal(0, 1, (bs, n_in)).astype(floatX)
in_pred_mat[:, 2] = 0
print tg_mat[3, 0]
print f_pred(in_pred_mat)[3, 0]
print f_pred(in_pred_mat)[3, 0]