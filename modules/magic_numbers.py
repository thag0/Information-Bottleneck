import json

magic_numbers = {# hiperparâmetros
    'epochs': 150,
    'tam_teste': 0.3,
    'tam_lote': 64,
    'num_bins': 30,
    'flat_mnist_input': True,
    'normalize_dataset_out': False,
    'tishby_dataset_len': 1024 * 4
}

def save_mn_config(mn: dict, filename: str):
    """
        Salva os valores do Magic Numbers em arquivo JSON.
    """
    
    with open(filename + '.json', "w") as f:
        json.dump(mn, f, indent = 4)

if __name__ == '__main__':
    print('magic_numbers.py é um módulo e não deve ser executado diretamente')