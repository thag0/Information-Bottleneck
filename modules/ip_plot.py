import numpy as np
import matplotlib.pyplot as plt

def _draw_information_plane(i_xt: np.ndarray, i_ty: np.ndarray, i_xy: float, epochs: int) -> None:
    """
        Função interna para desenhar o Plano da Informação.

        i_xt: np.ndarray
            Matriz 2D com I(X;T) para cada camada por época.

        i_ty: np.ndarray
            Matriz 2D com I(T;Y) para cada camada por época.

        i_xy: float
            Valor de I(X;Y).

        epochs: int
            Número de épocas de treino.
    """
    
    assert len(i_xt) == len(i_ty)

    plt.clf()
    plt.close('all')

    num_layers = len(i_xt)

    plt.figure(figsize=(10, 5))
    plt.xlabel(r'$I(X; T)$')
    plt.ylabel(r'$I(T; Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, epochs)]

    # Linhas horizontais entre camadas (por época)
    for epoch in range(epochs):
        for layer in range(num_layers - 1):
            plt.plot(
                [i_xt[layer, epoch], i_xt[layer + 1, epoch]],
                [i_ty[layer, epoch], i_ty[layer + 1, epoch]],
                color=colors[epoch], linestyle='-', linewidth=0.8, alpha=0.5, zorder=1
            )

    # Linhas verticais entre épocas (por camada)
    for i in range(num_layers):
        IXT = i_xt[i, :]
        ITY = i_ty[i, :]
        for epoch in range(epochs - 1):
            plt.plot(
                [IXT[epoch], IXT[epoch + 1]],
                [ITY[epoch], ITY[epoch + 1]],
                color=colors[epoch], linestyle='-', linewidth=0.8, alpha=0.5, zorder=2
            )

    # Pontos de cada época
    for i in range(num_layers):
        IXT = i_xt[i, :]
        ITY = i_ty[i, :]
        plt.scatter(IXT, ITY, marker = 'o', c = colors, s = 100, alpha = 1.0, zorder=10)

    # 4. Barra de cor e linha da informação mútua
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    fig, ax = plt.gcf(), plt.gca()
    cbar = fig.colorbar(sm, ax=ax, ticks=[])
    cbar.set_label('Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')

    plt.axhline(y=i_xy, color='red', linestyle=':', label=r'$I[X,Y]$')
    plt.title("Information Plane")
    plt.legend()

def plot_information_plane(i_xt: np.ndarray, i_ty: np.ndarray, i_xy: float, epochs: int) -> None:
    """
        Plota o Plano da Informação na tela.

        i_xt: np.ndarray
            Matriz 2D com I(X;T) para cada camada por época.

        i_ty: np.ndarray
            Matriz 2D com I(T;Y) para cada camada por época.

        i_xy: float
            Valor de I(X;Y).

        epochs: int
            Número de épocas de treino.    
    """

    print('Gerando gráfico')

    _draw_information_plane(i_xt, i_ty, i_xy, epochs)
    plt.show()

def save_information_plane(i_xt: np.ndarray, i_ty: np.ndarray, i_xy: float, epochs: int, filename: str) -> None:
    """
        Salva o Plano da Informação em um arquivo.

        i_xt: np.ndarray
            Matriz 2D com I(X;T) para cada camada por época.
        
        i_ty: np.ndarray
            Matriz 2D com I(T;Y) para cada camada por época.

        i_xy: float
            Valor de I(X;Y).

        epochs: int
            Número de épocas de treino.
    
        filename: str
            String contendo o diretório e nome do arquivo para salvamento.
    """
    print('Salvando gráfico')

    _draw_information_plane(i_xt, i_ty, i_xy, epochs)
    plt.tight_layout()
    plt.savefig(filename)

def save_train_info(train_info: dict, fl_base: str) -> None:
    """
        Salva os dados de treino (loss e accuracy) em arquivos PNG.   
        
        train_info: dict
            Dicionário do Keras retornado pelo método fit() do modelo.
        
        fl_base: str
            String contendo o diretório e nome base do arquivo para salvamento.
    """
    
    print('Salvando dados de treino')

    loss_filename = fl_base + '-loss'
    acc_filename = fl_base + '-accuracy'

    last_loss = train_info['loss'][-1]
    last_acc = train_info['accuracy'][-1]

    plt.clf()
    plt.close('all')

    plt.plot(train_info['loss'], label = 'Loss')
    plt.title(f'Loss ({last_loss:.8f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(loss_filename)

    plt.clf()
    plt.close('all')

    plt.plot(train_info['accuracy'], label = 'Accuracy')
    plt.title(f'Accuracy ({last_acc:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(acc_filename)

if __name__ == '__main__':
    print('ip_plot.py é um modulo e não deve ser executado diretamente.')