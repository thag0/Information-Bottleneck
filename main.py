from mi_utils import *
from data import *
from magic_numbers import MagicNumbers
from model import model

from keras.api.callbacks import LambdaCallback
import tensorflow as tf
import os

# def save_activations(model):
#     layer_outputs = [layer.output for layer in model.layers]
#     activation_model = tf.keras.Model(inputs = model.inputs, outputs = layer_outputs)
#     activations = activation_model.predict(X_train, batch_size = TAM_LOTE, verbose = 0)
#     global act_list
#     act_list.append(activations)

def save_activations(model):
    global act_list
    outputs = [layer.output for layer in model.layers if 'input' not in layer.name]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

    activations = activation_model.predict(X_train, batch_size=128, verbose=0)
    act_list.append(activations)

act_list = []

if __name__ == '__main__':
    os.system('cls')
    
    mn = MagicNumbers()

    NUM_EPOCAS = mn.epochs
    TAM_LOTE = mn.tam_lote
    NUM_BINS = mn.num_bins
    TAM_TESTE = mn.tam_teste
    NORM_DATASET_OUT = mn.normalize_dataset_out

    X_train, X_test, Y_train, Y_test = gen_data(TAM_TESTE, NORM_DATASET_OUT)

    input_shape = (784, )
    modelo = model(input_shape)
    NUM_LAYERS = len(modelo.layers)

    act_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: save_activations(modelo))
    
    print('treinado')
    result = modelo.fit(
        x = X_train,
        y = Y_train,
        epochs = NUM_EPOCAS,
        batch_size = TAM_LOTE,
        callbacks = [act_callback],
        verbose = 1
    )

    loss, acc = modelo.evaluate(X_test, Y_test, verbose = 0)
    print('Perda: ', loss, '\nAcurácia: ', acc)
    
    act_list = discretization(act_list, NUM_BINS, NUM_LAYERS, NUM_EPOCAS)

    i_xy = mutual_information(X_train, Y_train)
    i_xt, i_ty = information_plane(X_train, Y_train, act_list, NUM_LAYERS, NUM_EPOCAS)
    plot_information_plane(i_xt, i_ty, NUM_EPOCAS, i_xy)


