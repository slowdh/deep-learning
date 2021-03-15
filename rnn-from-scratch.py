import numpy as np


### utils ###
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

def softmax(x):
    e = np.exp(x)
    a = e / np.sum(e, axis=0)
    return a


### RNN ###

# forward path of RNN
# compute single forward time step
def rnn_cell_forward(xt, a_prev, parameters):
    # retrieve parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # caculate next activation
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # cache for back prop
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    # collects cache of each time step
    caches = []

    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize hidden state 'a' for all time steps and 'y' for all time steps
    a = np.zeros(shape=(n_a, m, T_x))
    y_pred = np.zeros(shape=(n_y, m, T_x))

    # loop over all time steps
    a_next = a0
    for t in range(T_x):
        xt = x[:,:,t]
        a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
        a[:,:,t] += a_next
        y_pred[:,:,t] += yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches