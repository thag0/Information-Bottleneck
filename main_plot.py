from modules.mi_utils import *
from modules.data import *
from modules.magic_numbers import magic_numbers as mn
from modules.model import mnist_model, tishby_model, mnist_conv_model
from modules.ip_plot import plot_information_plane, save_information_plane, save_train_info

from keras.api.callbacks import LambdaCallback
from keras.api.models import Sequential
import tensorflow as tf
import os
import sys
import gc

# desativar avisos
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
tf.get_logger().setLevel('ERROR')

def save_activations(model: Sequential):
    global act_list
    
    outputs = [layer.output for layer in model.layers]
    act_model = tf.keras.Model(inputs = model.inputs, outputs = outputs)

    activations = act_model.predict(X_train, batch_size = mn['tam_lote'], verbose = 0)
    act_list.append(activations)

def normalize_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
    return normalized_arr

act_list = [] # [epoch][layer][sample][neuron]

if __name__ == '__main__':
    os.system('cls')

    # Tishby
    input_shape, X_train, Y_train = generate_data(12, mn['tishby_dataset_len'])
    
    # MNIST
    # input_shape, X_train, X_test, Y_train, Y_test = mnist_data(mn['tam_teste'], mn['flat_mnist_input'])
    
    # MNIST Conv
    # input_shape, X_train, X_test, Y_train, Y_test = mnist_data(mn['tam_teste'], False)

    print('X: ', X_train.shape)
    print('Y: ', Y_train.shape)


    # modelo = mnist_conv_model(input_shape)
    # modelo = mnist_model(input_shape)
    modelo = tishby_model(input_shape)

    act_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: [
        save_activations(modelo),
    
        # Imprimir avanço do treinamento
        sys.stdout.write(f"\rÉpoca {epoch + 1}/{mn['epochs']}"),
        sys.stdout.flush()
    ])
    
    print('Treinando')
    result = modelo.fit(
        x = X_train,
        y = Y_train,
        epochs = mn['epochs'],
        batch_size = mn['tam_lote'],
        callbacks = [act_callback],
        verbose = 0,
    )

    loss, acc = modelo.evaluate(X_train, Y_train, verbose = 0)
    print('\n \nPerda: ', loss, '\nAcurácia: ', acc)

    act_list_2 = discretization(act_list, mn['num_bins'])

    I_XY = mutual_information(X_train, Y_train)
    I_XT, I_TY = information_plane(X_train, Y_train, act_list_2, len(modelo.layers), mn['epochs'])

    plot_information_plane(I_XT, I_TY, I_XY, mn['epochs'])