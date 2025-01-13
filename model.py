from keras.api.models import Sequential
from keras.api.layers import Dense, Input

def model(input_shape: tuple) -> Sequential:
    model =  Sequential([
        Input(shape = input_shape),
        Dense(15, activation = "relu"),
        Dense(15, activation = "relu"),
        Dense(10, activation = "softmax"),
    ])

    model.compile(
        optimizer = "sgd",
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
    )

    return model