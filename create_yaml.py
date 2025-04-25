import os
import yaml


def create_data_yaml(file_path):
    ## Ajouter 1 Classe : Crack
    data = {
        'train': r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\images',
        'val': r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\images',
        'nc': 7,
        'names': ['crack','crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    }
    with open(file_path, 'w') as f:
        yaml.dump(data, f)
    print(f"data.yaml created at {file_path}")
    