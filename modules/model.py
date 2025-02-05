from keras.api.models import Sequential
from keras.api.optimizers import SGD
from keras.api.layers import Dense, Input

# Modelo Tishiby
#   - MLP;
#   - Atv tanh, atv saida sigmoid
#   - Perda: binary_crossentropy
#   - Unidades: 12 - 10 - 8 - 6 - 4 - 2
#   - Dataset sintetico

def model(input_shape: tuple[int]) -> Sequential:

    model =  Sequential([
        Input(input_shape),
        Dense(10, activation = "tanh"),
        Dense( 7, activation = "tanh"),
        Dense( 5, activation = "tanh"),
        Dense( 4, activation = "tanh"),
        # Dense( 3, activation = "tanh"),
        # Dense( 2, activation = "tanh"),
        Dense( 1, activation = "sigmoid"),
    ])

    # model =  Sequential([
    #     Input(input_shape),
    #     Dense(10, activation = "tanh"),
    #     Dense( 8, activation = "tanh"),
    #     Dense( 6, activation = "tanh"),
    #     Dense( 4, activation = "tanh"),
    #     Dense( 2, activation = "tanh"),
    #     Dense( 1, activation = "sigmoid"),
    # ])

    model =  Sequential([
        Input(input_shape),
        Dense(25, activation = "tanh"),
        Dense(20, activation = "tanh"),
        Dense(15, activation = "tanh"),
        Dense(10, activation = "softmax"),
    ])

    # model.compile(
    #     optimizer = SGD(0.00001, 0.9999),
    #     loss = "binary_crossentropy",
    #     metrics = ['accuracy']
    # )

    model.compile(
        optimizer = SGD(0.000001, 0.999),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
    )

    return model

if __name__ == '__main__':
    print('model.py é um módulo e não deve ser executado diretamente')