import numpy as np


# get data from txt file
def get_and_preprocess_data():
    with open('tf2/datasets/dinos.txt', 'r') as dino:
        data = dino.read()
    data = data.lower()
    splitted = data.split('\n')
    np.random.shuffle(splitted)

    chars = sorted(list(set(data)))
    data_size, char_size = len(data), len(chars)
    print(f'num of total characters: {data_size}, num of unique chars: {char_size}')
    # -> num of total characters: 19909, num of unique chars: 27

    # tokenizing chars
    char_to_idx = {char:i for i, char in enumerate(chars)}
    idx_to_char = {i:char for i, char in enumerate(chars)}
    return splitted, char_to_idx, idx_to_char

def generate_data_pair_from_word(word, char_to_idx):
    x = [None] + [char_to_idx[c] for c in word]
    y = x[1:] + [char_to_idx['\n']]
    return x, y

# build rnn model!
# utils
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

def softmax(x):
    e_x = np.exp(x - np.max(x))
    a = e_x / np.sum(e_x, axis=0)
    return a

def one_hot_encoding(idx, vocab_size=27):
    x = np.zeros((vocab_size, 1))
    if idx is not None:
        x[idx] = 1
    return x

def initialize_parameters(na, nx, ny):
    Wax = np.random.randn(na, nx) * 0.01
    Waa = np.random.randn(na, na) * 0.01
    Wya = np.random.randn(ny, na) * 0.01
    b = np.zeros((na, 1))
    by = np.zeros((ny, 1))

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b) # hidden state update
    yhat_t = softmax(np.dot(Wya, a_next) + by)
    return a_next, yhat_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] -= lr * gradients['dWax']
    parameters['Waa'] -= lr * gradients['dWaa']
    parameters['Wya'] -= lr * gradients['dWya']
    parameters['b']  -= lr * gradients['db']
    parameters['by']  = lr * gradients['dby']
    return parameters

def rnn_forward(x_train, y_train, a0, parameters, vocab_size=27):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0

    for t in range(len(x_train)):
        x[t] = one_hot_encoding(x_train[t])
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        loss -= np.log(y_hat[t][y_train[t]])

    cache = (y_hat, a, x)
    return loss, cache

def rnn_backward(x_train, y_train, parameters, cache):
    gradients = {}
    y_hat, a, x = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # initialize as a same shape of original weights
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    # backprop through time
    for t in reversed(range(len(x_train))):
        dy = np.copy(y_hat[t])
        dy[y_train[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

    return gradients, a

# implement gradient clipping to prevent gradient exploding
def clip(gradients, abs_max_value):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for g in (dWaa, dWax, dWya, db, dby):
        np.clip(g, -abs_max_value, abs_max_value, out=g)
    return gradients

# implement sampling
def sample(parameters, char_to_idx, limit=50):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # initialize first input as zeros
    x = np.zeros(shape=(vocab_size, 1))
    a_prev = np.zeros(shape=(n_a, 1))

    indices = []
    newline_char = char_to_idx['\n']
    for c in range(limit):
        # single forward path in RNN model
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        y_hat = softmax(np.dot(Wya, a) + by)
        y_hat = np.ravel(y_hat)
        # sample to probability distribution
        idx = np.random.choice(vocab_size, p=y_hat)
        indices.append(idx)
        if idx == newline_char:
            break
        # updating x and a_prev
        x.fill(0)
        x[idx, :] = 1
        a_prev = a
    return indices

def decode_indices(lst_indices, idx_to_char):
    s = [idx_to_char[i] for i in lst_indices]
    if s[-1] == '\n':
        s = s[:-1]
    return ''.join(s)

# one step of gradient update
def step(x_train, y_train, a_prev, parameters, learning_rate=0.01):
    loss, cache = rnn_forward(x_train, y_train, a_prev, parameters)
    gradients, a = rnn_backward(x_train, y_train, parameters, cache)
    gradients = clip(gradients, 5)
    update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(x_train) - 1]

# train language model!

def model(examples, idx_to_char, char_to_idx, na=50, num_iterations=35000):
    nx = ny = len(idx_to_char)
    parameters = initialize_parameters(na, nx, ny)
    a_prev = np.zeros((na, 1))
    num_examples = len(examples)

    for i in range(num_iterations):
        idx = i % num_examples
        x, y = generate_data_pair_from_word(examples[idx], char_to_idx)
        loss, gradients, a_last = step(x, y, a_prev, parameters, learning_rate=0.01)

        if i % 2000 == 0:
            print('\n' + f'Iteration: {i}, Loss: {loss}')
            for j in range(5):
                sampled = sample(parameters, char_to_idx)
                print('    ' + decode_indices(sampled, idx_to_char))
    return parameters

# lets test result!
examples, char_to_idx, idx_to_char = get_and_preprocess_data()
parameters = model(examples, idx_to_char, char_to_idx, na=50, num_iterations=100000)


"""
Iteration: 0, Loss: [46.1434397]
    
    yoysyrgnriizbupabnppjsjxxrkcp
    jecjfqmubeco
    oxaf
    qtxmnqetdwuggnfmfoenow
    
    
...


Iteration: 98000, Loss: [22.94133864]
    kinophosaurus
    altodratorator
    sterraptor
    gayzodonus
    inoveia
"""
