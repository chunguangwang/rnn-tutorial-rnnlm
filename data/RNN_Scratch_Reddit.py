import random
import zipfile
import itertools
import operator
import csv
from datetime import datetime
import random
import sys
import nltk
import numpy as np
nltk.download('punkt')

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file…")
with open('reddit-comments-2015-08.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))
sentencesPartial = sentences[5000:25000]
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentencesPartial]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is ‘%s’ and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: ‘%s’" % sentences[0])
print("\nExample sentence after Pre-processing: ‘%s’" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])






## initialize parameters
class RNNNumpy():
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 5):
        # assign instance variable
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # random initiate the parameters
        self.U = np.loadtxt('RNNModel_U').view(float) #np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.loadtxt('RNNModel_V').view(float) #np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.loadtxt('RNNModel_W').view(float) #np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


## 1. forward propagation

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def forward_propagation(self, x):
    # total num of time steps, len of vector x
    T = len(x)
    # during forward propagation, save all hidden stages in s, S_t = U .dot x_t + W .dot s_{t-1}
    # we also need the initial state of s, which is set to 0
    # each time step is saved in one row in s，each row in s is s[t] which corresponding to an rnn internal loop time
    s = np.zeros((T+1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # output at each time step saved as o, save them for later use
    o = np.zeros((T, self.word_dim))
    for t in np.arange(T):
        # we are indexing U by x[t]. it is the same as multiplying U with a one-hot vector
        s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation





def predict(self, x):
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis = 1)

RNNNumpy.predict = predict

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print(o.shape)
print(o)

predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)




## 2. calculate the loss
'''
the loss is defined as
L(y, o) = -\frac{1}{N} \sum_{n \in N} y_n log(o_n)
'''
def calculate_total_loss(self, x, y):
    L = 0
    # for each sentence ...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # we only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L

def calculate_loss(self, x, y):
    # divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x, y)/N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

print("Expected Loss for random prediction: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(X_train[:100], y_train[:100]))





## 3. BPTT
'''
1. we nudge the parameters into a direction that reduces the error. the direction is given by the gradient of the loss: \frac{\partial L}{\partial U}, 
\frac{\partial L}{\partial V}, \frac{\partial L}{\partial W}
2. we also need learning rate: which indicated how big of a step we want to make in each direction
Q: how to optimize SGD using batching, parallelism and adaptive learning rates.

RNN BPTT: because the parameters are shared by all time steps in the network, the gradient at each output depends not only on the calculations of the
current time step, but also the previous time steps.
'''

def bptt(self, x, y):
    T = len(y)
    # perform forward propagation
    o, s = self.forward_propagation(x)
    # we will accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1   # it is y_hat - y
    # for each output backwards ...
    for t in np.arange(T):
        dLdV += np.outer(delta_o[t], s[t].T)    # at time step t, shape is word_dim * hidden_dim
        # initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] * s[t]))
        # backpropagation through time (for at most self.bptt_truncate steps)
        # given time step t, go back from time step t, to t-1, t-2, ...
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print("Backprogation step t=%d bptt step=%d" %(t, bptt_step))
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] += delta_t
            # update delta for next step
            dleta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] * s[bptt_step-1])
    return [dLdU, dLdV, dLdW]

RNNNumpy.bptt = bptt






### 3.1 gradient checking
'''
verify the gradient by its definition:
\frac{\partial{L}}{\partial{\theta}} = \lim_{h \propto 0} \frac{J(\theta + h) - J(\theta - h)}{2h}
'''
def gradient_check(self, x, y, h = 0.001, error_threshold = 0.01):
    # calculate the gradient using backpropagation
    bptt_gradients = self.bptt(x, y)
    # list of all params we want to check
    model_parameters = ["U", "V", "W"]
    # gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # get the actual parameter value from model, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print("performing gradient check for parameter %s with size %d. " %(pname, np.prod(parameter.shape)))
        # iterate over each element of the parameter matrix, e.g. (0,0), (0,1)...
        it = np.nditer(parameter, flags = ['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # save the original value so we can reset it later
            original_value = parameter[ix]
            # estimate the gradient using (f(x+h) - f(x-h))/2h
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # reset parameter to the original value
            parameter[ix] = original_value
            # the gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate the relative error (|x - y|)/(|x|+|y|)
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # if the error is too large fail the gradient check
            if relative_error < error_threshold:
                print("Gradient check error: parameter = %s ix = %s" %(pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for parameter %s passed. " %(pname))

RNNNumpy.gradient_check = gradient_check

# grad_check_vocab_size = 8000
# np.random.seed(10)
# model = RNNNumpy(grad_check_vocab_size, 100, bptt_truncate = 1000)
# model.gradient_check([0,1,2,3], [1,2,3,4])






## 4. SGD implementation
'''
two step:
1. calculate the gradients and perform the updates for one batch
2. loop through the training set and adjust the learning rate
'''
### 4.1. perform one step of SGD
def numpy_sgd_step(self, x, y, learning_rate):
    dLdU, dLdV, dLdW = self.bptt(x, y)
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
RNNNumpy.sgd_step = numpy_sgd_step

### 4.2. outer SGD loop
'''
 - model: 
 - X_train:
 - y_train:
 - learning_rate:
 - nepoch:
 - evaluate loss_after:
'''
def train_with_sgd(model, X_train, y_train, learning_rate = 0.005, nepoch = 100, evaluate_loss_after = 5):
    # keep track of the losses so that we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss))
            # adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("setting learning rate to %f" %(learning_rate))
            sys.stdout.flush()
        # for each training example...
        for i in range(len(y_train)):
            # one sgd step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
model.sgd_step(X_train[10], y_train[10], 0.005)



np.random.seed(10)
model = RNNNumpy(vocabulary_size)
# losses = train_with_sgd(model, X_train, y_train, nepoch = 45, evaluate_loss_after = 1)
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    count = 0
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        count = count+1
        if count > 25:
            break
    sentence_str = [(index_to_word[x] if x < len(index_to_word)-1 else 'hello') for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 100
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print(" ".join(sent))


# print(X_train[7])
# predictions =model.predict(X_train[7])
# print(predictions.shape)
# print(predictions)
# np.savetxt('RNNModel_U', model.U, fmt='%f')
# np.savetxt('RNNModel_V', model.V, fmt='%f')
# np.savetxt('RNNModel_W', model.W, fmt='%f')
