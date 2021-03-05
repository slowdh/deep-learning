import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_and_preprocess_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.
    x_test = x_test[..., np.newaxis] / 255.
    return (x_train, y_train), (x_test, y_test)

def get_model(input_shape):
    model = Sequential([
        Conv2D(8, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(2),
        Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=5, batch_size=256)
    return history

def plot_learning_curves(history, metric):
    df = pd.DataFrame(history.history)
    acc_plot = df.plot(y=metric, title='Accuracy vs Epochs')
    acc_plot.set(xlabel="Epochs", ylabel=metric)

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_loss, test_acc

def predict_model(model, x_test, y_test):
    random_indices = np.random.choice(x_test.shape[0], 4)
    images = x_test[random_indices, ...]
    labels = y_test[random_indices, ...]
    predictions = model.predict(images)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=-0.2)

    for i, (prediction, image, label) in enumerate(zip(predictions, images, labels)):
        axes[i, 0].imshow(np.squeeze(image))
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 0].text(10., -1.5, f'Digit {label}')
        axes[i, 1].bar(np.arange(len(prediction)), prediction)
        axes[i, 1].set_xticks(np.arange(len(prediction)))
        axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")
    plt.show()


(x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist_data()
model = get_model(input_shape=(28, 28, 1))
history = train_model(model, x_train, y_train)
plot_learning_curves(history, 'loss')
evaluate_model(model, x_test, y_test)
predict_model(model, x_test, y_test, 4)