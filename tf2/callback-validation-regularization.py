import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback

# learn how to add batchnorm, regularization methods to layer
#   regularizer -> l1, l2, dropout
# and  how to feed validation data to the model when training

model = Sequential()
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2))

opt = tf.keras.optimizers.Adam(learning_rate=0.05)
model.compile(optimizer=opt, loss='mse', metrics=['mape'])

# add validation set. both expression is possible.
history = model.fit(inputs, targets, validation_split=0.2)
history = model.fit(intputs, targets, validation_data=(x_val, y_val))


# adding callbacks
# What is 'logs' in callback parameter? and it says...
# The logs dictionary stores the loss value, along with all of the metrics we are using at the end of a batch or epoch.
# returns the same thing as history.history dict

class TrainCallback(Callback):
    def on_train_begin(self, logs=None):
        # Do something at the start of training
        pass

    def on_train_batch_begin(self, batch, logs=None):
        # for example
        if batch %2 ==0:
            print('\n After batch {}, the loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        pass

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, mode='max')
history2 = model.fit(x_train, y_train, epochs=5, callbacks=[TrainCallback(), early_stopping])