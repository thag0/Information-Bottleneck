from keras.api.models import Sequential
from keras.api.optimizers import SGD
from keras.api.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
import json

def model_config(model: Sequential) -> dict:
    config = {}

    for i in range(len(model.layers)):
        layer = model.layers[i]
        key = 'layer ' + str(i) + " - " + layer.__class__.__name__

        if (isinstance(layer, Dense)):
            config[key] = {
                'units': layer.units,
                'activation': layer.activation.__name__
            }
            
        elif (isinstance(layer, Conv2D)):
            config[key] = {
                'filters': layer.filters,
                'kernel_size': layer.kernel_size,
                'activation': layer.activation.__name__
            }

        elif (isinstance(layer, MaxPooling2D)):
            config[key] = {
                'pool_size': layer.pool_size,
                'strides': layer.strides,
                'padding': layer.padding
            }

        elif (isinstance(layer, Flatten)):
            config[key] = {}

    return config

def save_model_config(model: Sequential, filename: str):
    config = model_config(model)
    
    with open(filename + '.json', "w") as f:
        json.dump(config, f, indent = 4)

def tishby_model(input_shape: tuple[int]) -> Sequential:
    """
        Modelo baseado no artigo de Tishby.
    """

    #   - MLP;
    #   - Atv tanh, atv saida sigmoid
    #   - Perda: binary_crossentropy
    #   - Unidades: 12 - 10 - 8 - 6 - 4 - 2
    #   - Dataset sintetico

    # model =  Sequential([
    #     Input(input_shape),
    #     Dense(10, activation = "tanh"),
    #     Dense( 7, activation = "tanh"),
    #     Dense( 5, activation = "tanh"),
    #     Dense( 4, activation = "tanh"),
    #     Dense( 3, activation = "tanh"),
    #     Dense( 2, activation = "tanh"),
    #     Dense( 1, activation = "sigmoid"),
    # ])

    model =  Sequential([
        Input(input_shape),
        Dense(10, activation = "tanh"),
        Dense( 8, activation = "tanh"),
        Dense( 6, activation = "tanh"),
        Dense( 4, activation = "tanh"),
        Dense( 2, activation = "tanh"),
        Dense( 1, activation = "sigmoid"),
    ])

    model.compile(
        optimizer = SGD(0.001, 0.5),
        loss = "binary_crossentropy",
        metrics = ['accuracy']
    )

    return model

def mnist_model(input_shape: tuple[int]) -> Sequential:
    
    # model =  Sequential([
    #     Input(input_shape),
    #     Dense(8, activation = "tanh"),
    #     Dense(8, activation = "tanh"),
    #     Dense(8, activation = "tanh"),
    #     Dense(8, activation = "tanh"),
    #     Dense(10, activation = "softmax"),
    # ])

    model =  Sequential([
        Input(input_shape),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(8, activation = "tanh"),
        Dense(10, activation = "softmax"),
    ])

    model.compile(
        optimizer = SGD(0.00001, 0.995),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
    )

    return model

def mnist_conv_model(input_shape: tuple[int]) -> Sequential:
    # Muito pesado em memória mesmo sendo um modelo pequeno

    model =  Sequential([
        Input(input_shape),
        Conv2D(8, (3, 3), activation = 'relu'),
        MaxPooling2D((2, 2)),
        Conv2D(8, (3, 3), activation = 'relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(10, activation = "tanh"),
        Dense(10, activation = "softmax"),
    ])

    model.compile(
        optimizer = SGD(0.000001, 0.999),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
    )

    return model

if __name__ == '__main__':
    print('model.py é um módulo e não deve ser executado diretamente')