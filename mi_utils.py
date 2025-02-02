import numpy as np
import matplotlib.pyplot as plt

def entropy(y: np.ndarray) -> float:
    unique, count = np.unique(y, return_counts=True, axis=0)
    prob = count / len(y)
    en = np.sum((-1)*prob*np.log2(prob))

    return en

def joint_entropy(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        min_size = min(x.shape[0], y.shape[0])
        x = x[:min_size]
        y = y[:min_size]

    yx = np.c_[y, x]

    return entropy(yx)

def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    return joint_entropy(y, x) - entropy(x)

def mutual_information(y: np.ndarray, x: np.ndarray) -> float:
    return entropy(y) - conditional_entropy(y, x)

def discretization(act_list, num_bins = 30) -> list:
    discretized = []

    for epoch_acts in act_list:
        epoch_discretized = []
    
        for layer_acts in epoch_acts:
            layer_acts = np.arctan(layer_acts)  # Aplicando a transformação arctan
            bins_edges = np.linspace(-1, 1, num_bins + 1)
            epoch_discretized.append(np.digitize(layer_acts, bins_edges))
    
        discretized.append(epoch_discretized)
    
    return discretized

def information_plane(x: np.ndarray, y: np.ndarray, act_list: list, num_layers: int, epochs: int):
    print("Calculando plano da informação")

    i_xt = np.zeros((num_layers, epochs))
    i_ty = np.zeros((num_layers, epochs))

    for epoch in range(epochs):
        for layer in range(num_layers):
            act = act_list[epoch][layer]
            i_xt[layer, epoch] = mutual_information(act, x)
            i_ty[layer, epoch] = mutual_information(act, y)

    return i_xt, i_ty

def plot_information_plane(i_xt: np.ndarray, i_ty: np.ndarray, epochs: int, i_xy: np.ndarray):
    print('Gerando gráfico')

    assert len(i_xt) == len(i_ty)

    max_index = len(i_xt)

    plt.figure(figsize = (10, 5))
    plt.xlabel(r'$I(X; T)$')
    plt.ylabel(r'$I(T; Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, epochs)]
    cmap_layer = plt.get_cmap('Greys')
    clayer = [cmap_layer(i) for i in np.linspace(0, 1, max_index)]

    for i in range(0, max_index):
        IXT = i_xt[i, :]
        ITY = i_ty[i, :]

        plt.plot(IXT,ITY,color=clayer[i],linestyle=None,linewidth=2,label='Camada {}'.format(str(i+1)))
        plt.scatter(IXT,ITY,marker='o',c=colors,s=200,alpha=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    fig, ax = plt.gcf(), plt.gca()  # Obtém a figura e o eixo atual
    cbar = fig.colorbar(sm, ax=ax, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')

    # plt.axhline(y = float(i_xy), color = 'red', linestyle = ':', label = r'$I[X,Y]$')

    plt.legend()
    plt.show()