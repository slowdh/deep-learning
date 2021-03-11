import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

# load data
x_train = np.load('tf2/datasets/dog-vs-cat/images_train.npy')
x_valid = np.load('tf2/datasets/dog-vs-cat/images_valid.npy')
x_test = np.load('tf2/datasets/dog-vs-cat/images_test.npy')
y_train = np.load('tf2/datasets/dog-vs-cat/labels_train.npy')
y_valid = np.load('tf2/datasets/dog-vs-cat/labels_valid.npy')
y_test = np.load('tf2/datasets/dog-vs-cat/labels_test.npy')

# display data
class_names = ['Dog', 'Cat']
choices = np.random.choice(x_train.shape[0], 10, replace=False)

fig, ax = plt.subplots(1, 10, figsize=(20, 2))
for i, j in enumerate(choices):
    ax[i].imshow(x_train[j])
    ax[i].set_title(class_names[y_train[j]])
    ax[i].set_axis_off()

# building banchmark model to compare with tranfered model!
def build_benchmark_model(input_shape):
    x = x_input = Input(shape=(input_shape))
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    benchmark_model = Model(x_input, x)
    benchmark_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    return benchmark_model

# get model
benchmark_model = build_benchmark_model(x_train.shape[1:])

# lets use early stopping callback for convenience.
earlystopping = tf.keras.callbacks.EarlyStopping(patience=3)

# fit model
benchmark_history = benchmark_model.fit(x=x_train, y=y_train, batch_size=128, epochs=15, validation_data=(x_valid, y_valid), callbacks=[earlystopping])

# plot learning curve

# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# ax[0].plot(benchmark_history.history['accuracy'])
# ax[0].set_title('Accuracy vs. epochs')


# evaluate benchmark model


# import pretrained MobileNetV2
mobilenet = tf.keras.applications.MobileNetV2()
mobilenet.summary()

# remove_head and build new model
def get_new_model(mobilenet):
    mobilenet.trainable = False
    model_input = mobilenet.input
    last_layer_output = mobilenet.get_layer(name='global_average_pooling2d').output

    x = Dense(32, activation='relu')(last_layer_output)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    new_model = Model(model_input, x)
    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    return new_model


# get model
new_model = get_new_model(mobilenet)

# lets use early stopping callback for convenience.
earlystopping = tf.keras.callbacks.EarlyStopping(patience=3)

# fit model
tranferred_history = new_model.fit(x=x_train, y=y_train, batch_size=128, epochs=15, validation_data=(x_valid, y_valid), callbacks=[earlystopping])