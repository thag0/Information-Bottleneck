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

def gen_tishby_data(num_samples: int, tam_teste: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera os dados sintéticos baseados no estudo de Tishby.
    
    Parâmetros:
        num_samples (int): Número total de amostras a serem geradas.
        tam_teste (float): Proporção do conjunto de testes.

    Retorna:
        tuple: (input_shape, X_train, X_test, y_train, y_test)
    """
    # Gerar 12 entradas binárias representando pontos em uma esfera 2D
    X = np.random.randint(0, 2, size=(num_samples, 12))

    # Aplicar uma função de combinação esférica (harmônicas esféricas)
    weights = np.random.randn(12)  # Pesos aleatórios para simular a função f(x)
    f_x = np.dot(X, weights)  # Simula a projeção sobre a esfera

    # Aplicar uma função sigmoidal para gerar rótulos probabilísticos
    theta = np.median(f_x)  # Definir o limiar de decisão
    gamma = 10  # Parâmetro de inclinação da sigmoide
    probs = 1 / (1 + np.exp(-gamma * (f_x - theta)))  # Probabilidades da sigmoide
    y = (probs > 0.5).astype(int)  # Binarizar a saída

    # Normalizar as entradas para o intervalo [0,1]
    X = X.astype('float32')

    # Codificar as saídas em one-hot encoding
    y = to_categorical(y, num_classes=2)

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tam_teste, random_state=42)

    input_shape = (12,)

    return input_shape, X_train, X_test, y_train, y_test