import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

def gen_data(tam_teste: float, flat_input: bool, normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

    if flat_input:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        input_shape = (X_train.shape[1], )
    else:
        input_shape = X_train[0]

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = tam_teste, random_state = 42)


    return (input_shape, X_train, X_test, y_train, y_test)

def generate_data(input_len = 12, samples = 4096) -> tuple[tuple[int], np.ndarray, np.ndarray]:
    '''
        Dataset do artigo de Tishby
    '''
    np.random.seed(42)
    X = np.random.choice([0, 1], size = (samples, input_len))

    f_x = np.sum(X, axis = 1)
    theta = np.median(f_x)

    gamma = 10
    P_Y_given_X = 1 / (1 + np.exp(-gamma * (f_x - theta)))
    Y = np.random.binomial(1, P_Y_given_X)

    input_shape = (input_len,)

    return input_shape, X.astype(np.float32), Y.astype(np.float32)

if __name__ == '__main__':
    print('data.py é um módulo e não deve ser executado diretamente')