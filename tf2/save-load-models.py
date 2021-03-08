import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint


model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10,)),
    Dense(1)
])
model.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Save model's weights with ModelCheckpoint
# tensorflow version, h5 version is possible
checkpoint = ModelCheckpoint('training/my_model.{epoch}-{batch}-{loss:04d}', save_freq='epoch', save_weights_only=True, verbose=1, )
model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint])

# manually save weights?
model.save_weights('my_model.h5')
# or entire model?
model.save('my_model.h5')

# Load model
new_model = tf.keras.models.load_model('my_model.h5')
# or load weights?
model.load_weights('keras_model.h5')