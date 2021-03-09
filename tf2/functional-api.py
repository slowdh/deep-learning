import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate


# Build model.with multiple inputs and outputs
# Use layer object as function, which calls input tensor and returns output tensor
inputs = Input(shape=(32, 1), name='inputs')
h = Conv1D(16, 5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
aux_inputs = Input(shape=(12,), name='aux_inputs')
h = Concatenate()([h, aux_inputs])
outputs = Dense(20, activation='sigmoid', name='outputs')(h)
aux_outputs = Dense(1, name='aux_outputs')(h)

model = Model(inputs=[inputs, aux_inputs], outputs=[outputs, aux_outputs])

# One way of compile /  fit model
model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1, 0.4], metrics=['accuracy'])
history = model.fit([x_train, x_aux], [y_train, y_aux], validation_split=0.2, epochs=20)

# More explicit way of compile / fit model
model.compile(loss={'outputs': 'binary_crossentropy', 'aux_outputs': 'mse'}, loss_weights={'outputs': 1, 'aux_outputs': 0.4}, metrics=['accuracy'])
history = model.fit({'inputs': x_train, 'aux_inputs': x_aux}, {'outputs': y_train, 'aux_outputs': y_aux}, validation_split=0.2, epochs=20)


# Tensor?
sample_var = tf.Variable([-1, 2], dtype=tf.float32, name='my_var')
sample_var.assign([3.5, -1.])

x = sample_var.numpy()

# Constant?
x_constant = tf.constant([1, -1])
zero = tf.zeros((2, 2))
one = tf.ones((2, 2))
eye = tf.eye(3)

# Tensor Operations?
# Expanding dims
t1 = tf.expand_dims(x_constant, 0)
t2 = tf.expand_dims(x_constant, 1)
# Squeezing
t1 = tf.squeeze(t1, 0)

# Math time! most or operations are overwritten already
matrix_multiplied = tf.matmul(zero, one)
plus = zero + one
minus = zero - one
multiply = zero * one
div = zero / one

# random?
tn = tf.random.normal(shape=(2, 2), mean=0, stddev=1.)
tu = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype='int32')


# Accessing model layers?
def get_model():
    inputs = Input(shape=(32, 1))
    h = Conv1D(16, 5, activation='relu', name='conv1_layer', trainable=False)(inputs)
    h = AveragePooling1D(3)(h)
    h = Flatten(name='flatten1')(h)
    outputs = Dense(20, activation='sigmoid')(h)

    model = Model(inputs, outputs)
    return model

sample = get_model()
print(sample.layers[1].weights)
# tensor output with kernal weight and bias
print(sample.layers[1].kernel)
print(sample.layers[1].bias.numpy())

# or access with name of layer
print(sample.get_layer(name='conv1_layer').weights[0].numpy())


# Transfer learning?
sample.trainable = False

flatten_output = sample.get_layer('flatten1').output
outputs = Dense(15, activation='softmax', name='new_softmax')(flatten_output)
new_model = Model(inputs=sample.input, outputs=outputs)