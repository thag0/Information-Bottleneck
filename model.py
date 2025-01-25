from keras.api.models import Sequential
from keras.api.optimizers import SGD
from keras.api.layers import Dense, Input

# Modelo Tishiby
#   - MLP;
#   - Atv tanh, atv saida sigmoid
#   - Unidades: 12 - 10 - 8 - 6 - 4 - 2
#   - Dataset


def model(input_shape: tuple) -> Sequential:
    model =  Sequential([
        Input(shape = input_shape),
        Dense(10, activation = "tanh"),
        Dense( 8, activation = "tanh"),
        Dense( 6, activation = "tanh"),
        Dense( 4, activation = "tanh"),
        Dense( 2, activation = "sigmoid"),
    ])

    model.compile(
        optimizer = SGD(0.00001, 0.99),
        loss = "mse",
        metrics = ['accuracy']
    )

    return model