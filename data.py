import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

def gen_data(tam_teste: float, normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = tam_teste, random_state = 42)

    return (X_train, X_test, y_train, y_test)