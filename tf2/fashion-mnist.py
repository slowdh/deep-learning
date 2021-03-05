import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fashion_mnist_data = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist_data.load_data()
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# rescale input data & add dummy color channel
x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# visualize images
idx = 6
img = x_train[idx]
label = labels[y_train[idx]]
plt.imshow(img)
plt.show()
print(f'label: {label}')

# build model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# fit model
history = model.fit(x_train, y_train, batch_size=256, epochs=8, verbose=1)
df = pd.DataFrame(history.history)
print(df.head())

# visualize loss on each epochs
loss_plot = df.plot(y='loss', title='loss vs epochs')
loss_plot.set(xlabel='epochs', ylabel='loss')

# evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"loss: {test_loss} \n acc: {test_acc}")

# make prediction on image
idx = 15
test_img = x_test[idx]
plt.imshow(test_img)
plt.show()
prediction = model.predict(test_img[np.newaxis, ...])
print(f"Predicted label: {labels[np.argmax(prediction)]}")
print(f"True label: {labels[y_test[idx]]}")