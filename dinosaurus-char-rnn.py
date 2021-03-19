import numpy as np


### utils ###
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

def softmax(x):
    e = np.exp(x)
    a = e / np.sum(e, axis=0)
    return a


# get data from txt file
data = open('tf2/datasets/dinos.txt', 'r').read()
data = data.lower()
chars = sorted(list(set(data)))
data_size, char_size = len(data), len(chars)
print(f'num of total characters: {data_size}, num of unique chars: {char_size}')
# -> num of total characters: 19909, num of unique chars: 27

# chars to idx, idx to chars
char_to_idx = {char:i for i, char in enumerate(chars)}
idx_to_char = {i:char for i, char in enumerate(chars)}

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
