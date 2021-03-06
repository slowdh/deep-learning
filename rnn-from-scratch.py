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

# RNN for all time steps
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
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


### LSTM ###

# forward path of LSTM
# single time step
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight
    bi = parameters["bi"]
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]

    # retrieve dimensions
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt), axis=0)

    # compute ft(forget_gate), it(update gate), cct (candidate value c_tilda t), c_next, ot(output gate), a_next
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    c_next = c_prev * ft + it * cct
    a_next = ot * np.tanh(c_next)
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache

# LSTM for all time steps
def lstm_forward(x, a0, parameters):
    caches = []

    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # initialize a, c, y (for all time steps) with zeros
    a = np.zeros(shape=(n_a, m, T_x))
    c = np.zeros(shape=(n_a, m, T_x))
    y_pred = np.zeros(shape=(n_y, m, T_x))

    # loop over all time steps
    c_next = np.zeros(shape=(n_a, m))
    a_next = a0
    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, c, caches
