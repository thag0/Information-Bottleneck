from modules.magic_numbers import magic_numbers as mn
from modules.magic_numbers import save_mn_config
from modules.mi_utils import *
from modules.data import *
from modules.model import *
from modules.ip_plot import plot_information_plane, save_information_plane, save_train_info

from keras.api.callbacks import LambdaCallback
import os
import sys
import gc

# desativar avisos
import warnings
warnings.filterwarnings("ignore", category = UserWarning)

def create_unique_dir(base_dir):
    """
        Cria novos diretórios para salvar os resultados.
    """

    if os.path.exists(base_dir):
        i = 1
        while os.path.exists(f"{base_dir}{i}"):
            i += 1
        new_dir = f"{base_dir}{i}"

    else:
        new_dir = base_dir
    
    os.makedirs(new_dir)
    
    return new_dir

def get_save_dir() -> str:
    # Tishby
    # dir_base = "./results/new/tishby/12-10-8-6-4-2-1/"
    dir_base = "./results/new/tishby/12-10-7-5-4-3-2-1/"
    # dir_base = "./results/new/tishby/12-10-7-5-4-3-2-1 (relu)/"

    # MNIST
    # dir_base = "./results/new/mnist/784-8-8-8-10/"
    # dir_base = "./results/new/mnist/784-8-8-8-8-8-8-10/"
    # dir_base = "./results/new/mnist/784-8-8-8-8-8-8-10 (relu)/"

    # MNIST Conv
    # dir_base = "./results/new/mnist-conv/"

    return dir_base

def get_data() -> tuple[tuple[int], np.ndarray, np.ndarray]:
    # Tishby
    input_shape, X_train, Y_train = generate_data(12, mn['tishby_dataset_len'])
    
    # MNIST
    # input_shape, X_train, X_test, Y_train, Y_test = mnist_data(mn['tam_teste'], mn['flat_mnist_input'])
    
    # MNIST Conv
    # input_shape, X_train, X_test, Y_train, Y_test = mnist_data(mn['tam_teste'], False)

    return input_shape, X_train, Y_train

def get_model(input_shape: tuple[int]) -> Sequential:
    model = tishby_model(input_shape)
    # model = mnist_model(input_shape)
    # model = mnist_conv_model(input_shape) # pesado, necessario +16GB de RAM

    return model

def main():
    act_list = [] # [epoch][layer][sample][neuron]

    iterations = 2
    
    dir_base = get_save_dir()
    input_shape, X_train, Y_train = get_data()

    for iteration in range(iterations):
        # limpar cache
        act_list.clear()
        gc.collect()

        model = get_model(input_shape)

        act_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: [
            save_activations(model, act_list, X_train, mn['tam_lote']),

            # Imprimir avanço do treinamento
            sys.stdout.write(f"\rÉpoca {epoch + 1}/{mn['epochs']}"),
            sys.stdout.flush()
        ])

        os.system('cls')
        print(f'Iteração {iteration + 1}/{iterations}')

        print('Treinando')
        result = model.fit(
            x = X_train,
            y = Y_train,
            epochs = mn['epochs'],
            batch_size = mn['tam_lote'],
            callbacks = [act_callback],
            verbose = 0,
        )

        loss, acc = model.evaluate(X_train, Y_train, verbose = 0)
        print('\n \nPerda: ', loss, '\nAcurácia: ', acc)

        act_list = discretization(act_list, mn['num_bins'])

        I_XY = mutual_information(X_train, Y_train) # Informação mútua da entrada com a saída do dataset
        I_XT, I_TY = information_plane(X_train, Y_train, act_list, len(model.layers), mn['epochs']) # Informação mútua da entrada/saída com as ativações

        iteration_dir = create_unique_dir(dir_base)
        dir_ip = os.path.join(iteration_dir, "info-plane")
        dir_train = os.path.join(iteration_dir, "train")
        dir_model_config = os.path.join(iteration_dir, "model-config")
        dir_mn_config = os.path.join(iteration_dir, "magic-numbers")

        save_information_plane(I_XT, I_TY, I_XY, mn['epochs'], dir_ip)
        save_train_info(result.history, dir_train)
        save_model_config(model, dir_model_config)
        save_mn_config(mn, dir_mn_config)

if __name__ == '__main__':
    os.system('cls')
    main()