from modules.mi_utils import *
from modules.data import *
from modules.magic_numbers import magic_numbers as mn
from modules.model import model
from modules.ip_plot import plot_information_plane

from keras.api.callbacks import LambdaCallback
from keras.api.models import Sequential
import tensorflow as tf
import os

# desativar avisos
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
tf.get_logger().setLevel('ERROR')

def save_activations(model: Sequential):
    global act_list
    
    outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

    activations = activation_model.predict(X_train, batch_size=mn.tam_lote, verbose=0)
    act_list.append(activations)

def normalize_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
    return normalized_arr

act_list = [] # [epoch][layer][sample][neuron]

if __name__ == '__main__':
    os.system('cls')
    
    # input_shape, X_train, X_test, Y_train, Y_test = gen_data(mn['flat_mnist_input'], mn.flat_input, mn.normalize_dataset_out)
    input_shape, X_train, Y_train = generate_data(12, mn['tishby_dataset_len'])

    print('X: ', X_train.shape)
    print('Y: ', Y_train.shape)

    modelo = model(input_shape)
    act_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: save_activations(modelo))
    
    print('treinando')
    result = modelo.fit(
        x = X_train,
        y = Y_train,
        epochs = mn['epochs'],
        batch_size = mn['tam_lote'],
        callbacks = [act_callback],
        verbose = 0,
    )

    loss, acc = modelo.evaluate(X_train, Y_train, verbose = 0)
    print('Perda: ', loss, '\nAcurácia: ', acc)

    act_list_2 = discretization(act_list, mn['num_bins'])

    i_xy = mutual_information(X_train, Y_train)
    i_xt, i_ty = information_plane(X_train, Y_train, act_list_2, len(modelo.layers), mn['epochs'])

    plot_information_plane(i_xt, i_ty, mn['epochs'], i_xy)