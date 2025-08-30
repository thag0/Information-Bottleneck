import numpy as np
import matplotlib.pyplot as plt
import sys

def entropy(y: np.ndarray) -> float:
    """
        Calcula a Entropia de Shannon.
    """
    
    _, count = np.unique(y, return_counts=True, axis=0)
    prob = count / len(y)
    e = np.sum((-1)*prob*np.log2(prob))

    return e

def joint_entropy(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]: # Nivelar a quantidade de amostras pro conjunto menor
        min_size = min(x.shape[0], y.shape[0])
        x = x[:min_size]
        y = y[:min_size]

    yx = np.c_[y, x]

    return entropy(yx)

def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    return joint_entropy(y, x) - entropy(x)

def mutual_information(y: np.ndarray, x: np.ndarray) -> float:
    return entropy(y) - conditional_entropy(y, x)

def discretization(act_list: list[list], num_bins = 30) -> list[list]:
    print('Discretizando ativações')
    
    discretized = []

    for epoch_acts in act_list:
        epoch_discretized = []
    
        for layer_acts in epoch_acts:
            layer_acts = np.arctan(layer_acts)  # Aplicando a transformação arctan
            bins_edges = np.linspace(-1, 1, num_bins + 1)
            epoch_discretized.append(np.digitize(layer_acts, bins_edges))
    
        discretized.append(epoch_discretized)
    
    return discretized

def information_plane(x: np.ndarray, y: np.ndarray, act_list: list, num_layers: int, epochs: int, logs: bool = True, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    if logs: print("\nCalculando plano da informação")

    i_xt = np.zeros((num_layers, epochs))
    i_ty = np.zeros((num_layers, epochs))

    for epoch in range(epochs):
        if logs:
            sys.stdout.write(f"\rÉpoca {epoch + 1}/{epochs}")  # Avanço da transformação
            sys.stdout.flush()

        for layer in range(num_layers):
            act = act_list[epoch][layer]
            i_xt[layer, epoch] = mutual_information(act, x)
            i_ty[layer, epoch] = mutual_information(act, y)

    if logs: print()  # Retomar na linha de baixo

    return i_xt, i_ty

if __name__ == '__main__':
    print('mi_utils.py é um módulo e não deve ser executado diretamente')