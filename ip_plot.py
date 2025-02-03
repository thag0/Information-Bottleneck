import numpy as np
import matplotlib.pyplot as plt

def plot_information_plane(i_xt: np.ndarray, i_ty: np.ndarray, epochs: int, i_xy: float):
    print('Gerando gráfico')

    assert len(i_xt) == len(i_ty)

    num_layers = len(i_xt)

    plt.figure(figsize = (10, 5))
    plt.xlabel(r'$I(X; T)$')
    plt.ylabel(r'$I(T; Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, epochs)]
    cmap_layer = plt.get_cmap('Greys')
    clayer = [cmap_layer(i) for i in np.linspace(0, 1, num_layers)]

    # Conectar camadas 
    for epoch in range(epochs):
        for layer in range(num_layers - 1):
            plt.plot(
                [i_xt[layer, epoch], i_xt[layer + 1, epoch]],  
                [i_ty[layer, epoch], i_ty[layer + 1, epoch]],  
                color=colors[epoch], linestyle = '-', linewidth = 0.8, alpha = 0.3
            )

    for i in range(num_layers):
        IXT = i_xt[i, :]
        ITY = i_ty[i, :]

        plt.scatter(IXT, ITY, marker = 'o', c = colors, s = 100, alpha = 1)

        # Conectar épocas
        for epoch in range(epochs - 1):
            plt.plot(
                [IXT[epoch], IXT[epoch + 1]],
                [ITY[epoch], ITY[epoch + 1]],
                color = colors[epoch], linestyle = '-', linewidth = 0.8, alpha = 0.5
            )

    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = 1))
    fig, ax = plt.gcf(), plt.gca()
    cbar = fig.colorbar(sm, ax = ax, ticks = [])
    cbar.set_label('Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform = cbar.ax.transAxes, va = 'top', ha = 'center')
    cbar.ax.text(0.5, 1.0, str(epochs), transform = cbar.ax.transAxes, va = 'bottom', ha = 'center')

    plt.axhline(y = i_xy, color = 'red', linestyle = ':', label = r'$I[X,Y]$')

    plt.title("Information Plane")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print('ip_plot.py é um modulo e não deve ser executado diretamente.')