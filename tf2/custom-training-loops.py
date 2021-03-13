import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# How to use GradientTape?
x = tf.constant([0, 1, 2, 3], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x ** 2)
    z = tf.math.exp(y)
    dz_dy, dz_dx = tape.gradient(z, [y, x])


# lets play with this!
# define custom layer
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.m = self.add_weight(shape=(1,), initializer='random_normal')
        self.b = self.add_weight(shape=(1,), initializer='zeros')

    def call(self, inputs):
        x = self.m * inputs + self.b
        return x

# define custom loss
def SquaredError(y_pred, y_true):
    error = tf.reduce_mean((y_pred - y_true) ** 2)
    return error

# generate noisy data
def make_noise_data(m, b, n=20):
    x = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(n,), stddev=0.1)
    y = m * x + b + noise
    return x, y

x_train, y_train = make_noise_data(1, 2)

# check my custom layer works
linear_regression = LinearLayer()
print(SquaredError(linear_regression(x_train), y_train))

# train and plot model
learning_rate = 0.05
steps = 25

for i in range(steps):
    with tf.GradientTape() as tape:
        prediction = linear_regression(x_train)
        loss = SquaredError(prediction, y_train)

    gradients = tape.gradient(loss, linear_regression.trainable_variables)
    linear_regression.m.assign_sub(learning_rate * gradients[0])
    linear_regression.b.assign_sub(learning_rate * gradients[1])

    print(f"Step: {i}, Loss: {loss.numpy()}")


# visualize the result!
plt.scatter(x_train, y_train)
plt.plot(x_train, linear_regression(x_train))
plt.show()