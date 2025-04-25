# 🧠 Image Detection Tools for YOLO Training & Debugging

Ce repo contient une suite d'outils Python pour préparer, augmenter et visualiser des jeux de données d'images annotées pour la détection d’objets avec le format **YOLO**.  
Les outils sont orientés vers un usage pratique dans un contexte de recherche ou d'industrie (ex : détection de défauts métallurgiques).

---




## 📦 Contenu du dépôt

| Script                         | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `augmentation.py`              | Applique des transformations (rotation, flip, HSV, etc.) sur un dataset (images + annotations VOC) en conservant la cohérence des bounding boxes. |
| `convert_voc_to_text.py`       | Convertit les annotations **PASCAL VOC (XML)** vers le format **YOLO (txt)**. |
| `draw_bbox_check_img_yolo.py`  | Affiche et sauvegarde les images avec leurs bounding boxes YOLO pour vérification visuelle. |

---

## 📁 Structure des dossiers attendue



Dataset/

├── train/

│   ├── images/

│   ├── labels/ (YOLO format)

│   ├── annotations/ (PASCAL VOC format)

├── validation/

│   ├── images/

│   ├── labels/

---

## 🔧 Utilisation

### 1. ⚙️ Augmenter un dataset (PASCAL VOC)

```python
from augmentation import augment_dataset

augment_dataset(
    input_img_dir='Dataset/train/images/crack',
    input_xml_dir='Dataset/train/annotations',
    output_img_dir='Dataset/train_augmented_crack/images',
    output_xml_dir='Dataset/train_augmented_crack/annotations'
)
```

Transformations appliquées :
- Rotation
- Flip horizontal
- Translation
- Échelle
- Changement HSV
- Perspective

### 2. 🔁 Conversion XML ➜ YOLO

python convert_voc_to_text.py

## ⚠️ Assurez-vous de bien configurer :
- Le chemin des dossiers XML en entrée
- Le dossier YOLO en sortie
- La liste des classes (classes = [...])

### 3. 👁️ Visualisation des bounding boxes YOLO

python draw_bbox_check_img_yolo.py

Ce script lit les .txt au format YOLO et dessine les bounding boxes sur les images, avec les labels, pour vérification manuelle.
## 📚 Dépendances

Installez les packages nécessaires avec :

pip install -r requirements.txt

Exemple de requirements.txt :

opencv-python
albumentations
numpy

## 🧪Exemple de classes utilisées

classes = ['crack', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

📜 Licence

Ce projet est sous licence MIT – voir le fichier LICENSE pour plus d’informations.
