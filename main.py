from mi_utils import *
import os

import keras.api.backend as K
from keras.api.models import Sequential
from keras.api.layers import Dense, Input
from keras.api.callbacks import LambdaCallback

def model() -> Sequential:
    model =  Sequential([
        Input((5,)),
        Dense(10, activation = "sigmoid"),
        Dense(10, activation = "softmax"),
    ])

    model.compile(
        optimizer = "sgd",
        loss = "mse"
    )

    return model

def save_activations(model) -> list:

    outs = [layer.outuput for layer in model.layers]
    functors = [K.function([model.input],[out]) for out in outs]
    layer_acts = [f([X_train]) for f in functors]
    act_list.append(layer_acts)

act_list = []

if __name__ == '__main__':
    os.system('cls')
    
    NUM_EPOCAS = 4
    TAM_LOTE = 28
    NUM_BINS = 30

    X_train, X_test, Y_train, Y_test = ...

    modelo = model()
    act_callback = LambdaCallback(on_epoch_end = lambda batch, logs: save_activations(modelo))
    
    result = modelo.fit(
        x = X_train,
        y = Y_train,
        epochs= NUM_EPOCAS,
        batch_size = TAM_LOTE,
        callbacks= [act_callback]
    )

    act_list = discretization(act_list, NUM_BINS, NUM_EPOCAS)
    info_plane = information_plane(X_train, Y_train, act_list, modelo.layers, NUM_EPOCAS)



