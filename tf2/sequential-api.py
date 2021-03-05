import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# simple neural network
simple_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu', name='layer_1'),
    Dense(16, activation='relu', name='layer_2'),
    Dense(10, activation='softmax', name='layer_3')
])
print(simple_model.summary())

# convolutional nn
conv_model = Sequential([
    Conv2D(16, 3, activation='relu', strides=2, padding='same', input_shape=(28, 28, 3), data_format='channels_last'),
    MaxPooling2D(3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
print(conv_model.summary())

# initialization methods
# use kernel_initializer, bias_initializer
test = Sequential([
    Dense(3, activation='relu', input_shape=(100), kernel_initializer='random_uniform', bias_initializer='zeros'),
    Dense(3, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))
])

# compile method
simple_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# this could be also written as
simple_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.075),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7), tf.keras.metrics.MeanAbsoluteError()]
)

# fit method
history = simple_model.fit(x=x_train, y=y_train, epochs=10, batch_size=16)

# evaluate and predict
loss, accuracy, mae = simple_model.evaluate(x_test, y_test)
pred = simple_model.predict(x_test)