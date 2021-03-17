from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np

# load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(start_char=1, oov_char=2, index_from=3)
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# for encoding and decoding
index_from = 3
imdb_word_to_idx = {word: index + index_from for word, index in imdb.get_word_index().items()}
imdb_idx_to_word = {idx: word for word, idx in imdb_word_to_idx.items()}

# decode samples!
idx = 0
classes = ["Negative", "Positive"]
decoded = [imdb_idx_to_word[i] for i in x_train[idx] if i > index_from]
label = classes[y_train[idx]]
print(f"Sentence: {' '.join(decoded)}")
print(f"Label: {label}")

# preprocess data
padded_x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300, padding='post',truncating='pre')
print(padded_x_train.shape) # now it is rectangular shape!

# for masking
padded_x_train = tf.expand_dims(padded_x_train, -1)
print(padded_x_train.shape) # quick shape check
masking_layer = tf.keras.layers.Masking(mask_value=0)
masked_x_train = masking_layer(padded_x_train)
print(masked_x_train._keras_mask)