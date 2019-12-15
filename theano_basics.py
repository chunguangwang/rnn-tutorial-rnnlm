import theano
from theano import tensor as T
import numpy as np

W_values = np.random.uniform((-1/3, 1/3), (3, 4))
b

W = theano.shared(W_values) # we assume that ``W_values`` contains the
                            # initial values of your weight matrix

bvis = theano.shared(bvis_values)
bhid = theano.shared(bhid_values)

trng = T.shared_randomstreams.RandomStreams(1234)

def OneStep(vsample) :
    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
    hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
    return trng.binomial(size=vsample.shape, n=1, p=vmean,
                         dtype=theano.config.floatX)

sample = theano.tensor.vector()

values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values[-1], updates=updates)



a = theano.shared(1)
values, updates = theano.scan(lambda: {a: a+1}, n_steps=10)

b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c], updates=updates)

print(b)
print(c)
print(a.get_value())
