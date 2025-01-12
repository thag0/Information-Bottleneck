import numpy as np

def entropy(x):
    unique, count = np.unique(x, return_counts=True, axis=0)
    prob = count/len(x)
    en = np.sum((-1)*prob*np.log2(prob))
    
    return en

def joint_entropy(x, y):
    yx = np.c_[y, x]
    return entropy(yx)

def conditional_entropy(x, y):
    return joint_entropy(y, x) - entropy(x)

def mutual_information(x, y):
    return entropy(y) - conditional_entropy(x, y)

def discretization(act_list, bins: int, layers, epochs: int):
    n_bins = bins

    for layer in range (0, len(layers)):
        for epoch in range (0, epochs):
            bins = np.linspace(
                min(np.min(act_list[epoch][layer][0], axis = 1)),
                max(np.max(act_list[epoch][layer][0], axis = 1)),
                n_bins+1                
            )

            act_list[epoch][layer][0] = np.digitize(act_list[epoch][layer][0], bins)

    return act_list

def information_plane(x, y, act_list, layers, epochs: int) -> tuple:

    i_xt = np.zeros(len(layers), epochs)
    i_ty = np.zeros(len(layers), epochs)

    for layer in range (0, len(layers)):
        for epoch in range (0, epochs):
            i_xt[layer, epoch] = mutual_information(act_list[epoch][layer][0], x)
            i_ty[layer, epoch] = mutual_information(act_list[epoch][layer][0], y)

    return i_xt, i_ty

def plot_information_plane():
    ...
