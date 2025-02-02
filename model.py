from keras.api.models import Sequential
from keras.api.optimizers import SGD
from keras.api.layers import Dense, Input

# Modelo Tishiby
#   - MLP;
#   - Atv tanh, atv saida sigmoid
#   - Perda: cross_entropy
#   - Unidades: 12 - 10 - 8 - 6 - 4 - 2
#   - Dataset

def model(input_shape: tuple) -> Sequential:

    model =  Sequential([
        Input(input_shape),
        Dense(10, activation = "tanh"),
        Dense( 7, activation = "tanh"),
        Dense( 5, activation = "tanh"),
        Dense( 4, activation = "tanh"),
        Dense( 3, activation = "tanh"),
        Dense( 2, activation = "tanh"),
        Dense( 1, activation = "sigmoid"),
    ])

    model.compile(
        optimizer = SGD(0.00001, 0.996),
        loss = "binary_crossentropy",
        metrics = ['accuracy']
    )

    return model