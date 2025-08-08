import os


def assign_new_class_index(base_path, new_class_index):
    """
    Atualiza os índices de classe de 0 para 'new_class_index'
    em todos os arquivos de label dentro da estrutura train/val/test/labels em base_path.
    Assume que todos os labels de entrada têm o índice de classe 0.

    Args:
        base_path (str): O caminho para o diretório raiz do dataset
                         (onde estão as pastas 'train', 'valid', 'test').
        new_class_index (int): O novo valor do índice da classe para substituir o 0.
    """
    splits = ['train', 'valid', 'test']

    # Verifica se o novo_class_index é inteiro e não negativo
    if not isinstance(new_class_index, int) or new_class_index < 0:
        raise ValueError("new_class_index deve ser um número inteiro não negativo.")

    print(f"Iniciando a atualização de índices de classe de 0 para {new_class_index} em {base_path}...")

    for split in splits:
        labels_dir = os.path.join(base_path, split, 'labels')
        if not os.path.exists(labels_dir):
            print(f"Diretório de labels não encontrado: {labels_dir}. Pulando este split.")
            continue

        print(f"Processando labels em: {labels_dir}")
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_dir, filename)

                with open(filepath, 'r') as f:
                    lines = f.readlines()

                modified = False
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        try:
                            current_class_index = int(parts[0])
                            if current_class_index == 0:
                                parts[0] = str(new_class_index)
                                modified = True
                            updated_lines.append(' '.join(parts) + '\n')
                        except ValueError:
                            print(f"Aviso: Linha mal formatada no arquivo {filename}: '{line.strip()}'")
                            updated_lines.append(line)
                    else:
                        updated_lines.append(line)

                if modified:
                    with open(filepath, 'w') as f:
                        f.writelines(updated_lines)

    print(f"Atualização de índices de classe concluída: de 0 para {new_class_index}.")

# def merge_yaml_files():

# def merge_datasets():

if __name__ == "__main__":
    base_path = 'dataset\Head'
    new_class_index = 1 
    assign_new_class_index(base_path, new_class_index)