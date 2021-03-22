from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np

# load dataset
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(start_char=1, oov_char=2, index_from=3, num_words=num_words)
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# for encoding and decoding
index_from = 3
imdb_word_to_idx = {word: index + index_from for word, index in imdb.get_word_index().items() if index + index_from < num_words}
imdb_idx_to_word = {idx: word for word, idx in imdb_word_to_idx.items()}

# decode samples!
idx = 0
classes = ["Negative", "Positive"]
decoded = [imdb_idx_to_word[i] for i in x_train[idx] if i > index_from]
label = classes[y_train[idx]]
print(f"Sentence: {' '.join(decoded)}")
print(f"Label: {label}")

# preprocess data
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=None, padding='pre',truncating='pre', value=0)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=None, padding='pre', truncating='pre', value=0)
print(x_train.shape) # now it is rectangular shape!

# for masking
x_train = tf.expand_dims(x_train, -1)
print(x_train.shape) # quick shape check
masking_layer = tf.keras.layers.Masking(mask_value=0)
x_train = masking_layer(x_train)
print(masked_x_train._keras_mask)

# make embedding layer!
max_idx_value = max(imdb_idx_to_word.keys())
embedding_layer = tf.keras.layers.Embedding(input_dim=max_idx_value + 1, output_dim=16, mask_zero=True)

# what is rnn layer doing?
simple_rnn_layer = tf.keras.layers.SimpleRNN(units=16)
sequence = tf.constant([[[1, 1], [2, 2], [56, -100]]], dtype=tf.float32)
layer_output = simple_rnn_layer(sequence)
print(layer_output)
print(sequence.shape)

# build model!
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_idx_value + 1, output_dim=16, mask_zero=True),
    tf.keras.layers.LSTM(units=16),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# train model!
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=32, epochs=3)
