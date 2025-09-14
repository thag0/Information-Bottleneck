import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from modules.mi_utils import *
from modules.data import *
import os

def tishby_data(input_len=12, samples=4096):
    np.random.seed(42)
    X = np.random.choice([0, 1], size=(samples, input_len))

    f_x = np.sum(X, axis=1)
    theta = np.median(f_x) # limiar de separação

    gamma = 10
    P_Y_given_X = 1 / (1 + np.exp(-gamma * (f_x - theta)))
    Y = np.random.binomial(1, P_Y_given_X)

    input_shape = (input_len,)
    return input_shape, X.astype(np.float32), Y.astype(np.float32), f_x, theta

def generate_graphics():
    input_shape, X, Y, f_x, theta = tishby_data()

    # Cria figura com 2 colunas
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    num_examples = 10
    idxs = np.random.choice(len(X), num_examples, replace=False)
    hspace = 0.2  # espaçamento horizontal entre cada entrada

    for i, idx in enumerate(idxs):
        row = X[idx]
        y_val = int(Y[idx])
        # Adiciona deslocamento no eixo Y para separar linhas
        y_offset = num_examples - i - 1  
        for j, bit in enumerate(row):
            color = "black" if bit == 1 else "lightgray"
            axes[0].add_patch(
                plt.Rectangle((j*(1+hspace), y_offset), 1, 0.8, color=color)
            )
        # Rótulo da saída Y ao lado
        axes[0].text(len(row)*(1+hspace)+0.5, y_offset+0.4,
                    f"Y={y_val}", va="center", fontsize=10)

    axes[0].set_xlim(-0.5, input_shape[0]*(1+hspace)+2)
    axes[0].set_ylim(-0.5, num_examples)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Amostras")
    # ------------------------
    # 2) Histograma das somas
    # ------------------------
    axes[1].hist(f_x[Y==0], bins=np.arange(-0.5, input_shape[0]+1.5, 1),
                 alpha=0.6, label="Y=0")
    axes[1].hist(f_x[Y==1], bins=np.arange(-0.5, input_shape[0]+1.5, 1),
                 alpha=0.6, label="Y=1")
    axes[1].axvline(theta, color="red", linestyle="--", label=f"θ = {theta:.0f}")
    axes[1].set_xlabel("Soma dos bits")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title("Distribuição das somas")
    axes[1].legend()

    # fig.suptitle("Dataset de Tishby", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_mnist_samples(X, Y, n_rows=3, n_cols=6):
    """
    Plota amostras do dataset MNIST
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(0, len(X))
        img = X[idx].reshape(28, 28)
        label = np.argmax(Y[idx]) if Y.ndim > 1 else Y[idx]

        ax.imshow(img, cmap="gray")
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_tanh_vs_relu():
    x = np.linspace(-5, 5, 200)

    # Funções de ativação
    def tanh(x): return np.tanh(x)
    def relu(x): return np.maximum(0, x)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for ax, func, title, y_lim in zip(axs, [tanh(x), relu(x)], ['Tanh', 'ReLU'], [[-1.1, 1.1], [-5, 5]]):
        ax.axvline(x=0, color='dimgray', linewidth=1.5, linestyle='--')
        ax.axhline(y=0, color='dimgray', linewidth=1.5, linestyle='--')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(y_lim)
        ax.plot(x, func, linewidth=2.5, color='black')
    
    # # Sigmoid
    # axs[0].plot(x, sigmoid, color='black', linewidth=3.0)
    # axs[0].set_title("Sigmoid")
    # axs[0].grid(True, linestyle="--", alpha=0.6)
    # axs[0].set_ylim()
    # axs[0].axvline(x=0, color="dimgray", linewidth=1.5, linestyle="--")
    
    # # ReLU
    # axs[1].plot(x, relu, color='black', linewidth=3.0)
    # axs[1].set_title("ReLU")
    # axs[1].grid(True, linestyle="--", alpha=0.6)
    # axs[1].set_ylim()
    # axs[1].axvline(x=0, color="dimgray", linewidth=1.5, linestyle="--")

    plt.show()

def main():
    # input_shape, X_train, _, Y_train, _ = mnist_data(0.3, flat_input=False)
    # plot_mnist_samples(X_train, Y_train, 5, 10)

    plot_tanh_vs_relu()

if __name__ == '__main__':
    os.system('cls')
    main()

