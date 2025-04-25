import torch
from torchvision.models.detection import ssd300_vgg16
import os

# Chemin vers le modèle sauvegardé
model_path = r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\RESULTS\ssd20_04_2025\ssd_best_model.pt'

# Vérifie que le fichier existe
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Le fichier '{model_path}' n'existe pas.")

# Nombre de classes que ton modèle est censé prédire (à adapter !)
NUM_CLASSES = 1  # ⚠️ change ceci selon ton entraînement

# Reconstruit l'architecture du modèle
try:
    model = ssd300_vgg16(pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Modèle chargé avec succès via state_dict.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Image factice pour test
dummy_input = torch.randn(1, 3, 300, 300)

# Prédiction
try:
    with torch.no_grad():
        output = model(dummy_input)
    print("Prédiction réussie.")
    print("Sortie du modèle :", output)
except Exception as e:
    raise RuntimeError(f"Erreur lors de la prédiction : {e}")