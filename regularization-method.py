import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from models import SequentialModel
from layers import DenseLayer
from regularizer import *
from optimizer import *


# get data
np.random.seed(3)
x_train, y_train = sklearn.datasets.make_moons(n_samples=300, noise=.2)

# Visualize the data
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, s=40, cmap=plt.cm.Spectral)
x_train = x_train.T
y_train = y_train.reshape((1, y_train.shape[0]))


#non-regularized model
non_regularized = SequentialModel(layers=[
    DenseLayer(dim=20, activation='relu'),
    DenseLayer(dim=5, activation='relu'),
    DenseLayer(dim=1, activation='sigmoid')
])

costs = non_regularized.fit(X=x_train, Y=y_train, loss='binary_crossentropy', num_iterations=10000, learning_rate=0.075, print_status=True, print_freq=1000)


# L2 regularization
l2_reg= l2(0.1)

l2_regularized = SequentialModel(layers=[
    DenseLayer(dim=20, activation='relu', regularizer=l2_reg),
    DenseLayer(dim=5, activation='relu', regularizer=l2_reg),
    DenseLayer(dim=1, activation='sigmoid', regularizer=l2_reg)
])

l2_costs = l2_regularized.fit(X=x_train, Y=y_train, loss='binary_crossentropy', num_iterations=10000, learning_rate=0.075, print_status=True, print_freq=1000)


# Dropout regularization
dropout_reg = dropout(keep_prob=0.8)

dropout_regularized = SequentialModel(layers=[
    DenseLayer(dim=20, activation='relu', regularizer=dropout(0.5)),
    DenseLayer(dim=5, activation='relu', regularizer=dropout(0.9)),
    DenseLayer(dim=1, activation='sigmoid')
])

dropout_regularized_costs = dropout_regularized.fit(X=x_train, Y=y_train, optimizer=Adam(), loss='binary_crossentropy', num_iterations=10000, learning_rate=0.075, print_status=True, print_freq=1000)

