# 🧠 Image Detection Tools for YOLO Training & Debugging

Ce repo contient une suite d'outils Python pour préparer, augmenter et visualiser des jeux de données d'images annotées pour la détection d’objets avec le format **YOLO**.  
Les outils sont orientés vers un usage pratique dans un contexte de recherche ou d'industrie (ex : détection de défauts métallurgiques).

---




## 📦 Contenu du dépôt

| Fichier                         | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `augmentation.py`              | Applique des transformations d’augmentation (flip, rotation, HSV, etc.) sur des images + annotations PASCAL VOC. |
| `classes.csv`                  | Fichier CSV listant les classes utilisées dans le dataset. |
| `conda_fix.py`                 | Patch de compatibilité pour les environnements conda (ex. : `KMP_DUPLICATE_LIB_OK`). |
| `convert_voc_to_text.py`       | Convertit les fichiers d'annotation XML (VOC) en format texte YOLO. |
| `correct_label.py`             | Corrige ou ajuste des erreurs dans les annotations (label/bbox). |
| `create_yaml.py`               | Génère un fichier `data.yaml` compatible avec YOLOv5/YAML pour l'entraînement. |
| `data.yaml`                    | Fichier de configuration YAML décrivant le dataset pour l'entraînement YOLO. |
| `data_test.yaml`              | Variante de `data.yaml` pour tests rapides ou validations. |
| `draw_bbox_check_csv.py`       | Visualise les bounding boxes définies dans un fichier CSV. |
| `draw_bbox_check_img_voc.py`   | Dessine les bboxes depuis les annotations PASCAL VOC (.xml). |
| `draw_bbox_check_img_yolo.py`  | Dessine les bboxes YOLO (.txt) sur les images pour contrôle visuel. |
| `draw_yolo_architecture.py`    | Génère une visualisation de l’architecture d’un modèle YOLO (version embedded/test). |
| `generate_annotation_csv.py`   | Regroupe ou convertit les annotations en format CSV. |
| `inference_test_embedded.py`   | Teste un modèle embarqué (YOLO/SSD) en mode inférence sur un ensemble d’images. |
| `README.md`                    | Ce fichier. Documentation du projet. |
| `ssd_evaluate_metrics.py`      | Évalue un modèle SSD à l’aide de métriques type mAP, recall, etc. |
| `ssd_pt_inspection.py`         | Permet d’inspecter un modèle SSD `.pt` (structure, couches, paramètres). |
| `test_cuda.py`                 | Script de test pour vérifier si CUDA est bien disponible. |
| `train_model_ssd.py`           | Script d'entraînement d’un modèle SSD. |
| `train_model_yolo.py`          | Script d'entraînement d’un modèle YOLO (YOLOv5 ou dérivé). |
| `xml_transcipt_fix.py`         | Nettoie ou corrige des fichiers XML d’annotations corrompus. |
| `xml_transcript.py`            | Génère ou convertit des fichiers XML d’annotations à partir de formats bruts. |

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

## ⚠️ Assurez-vous de bien configurer :
- Le chemin des dossiers XML en entrée
- Le dossier YOLO en sortie
- La liste des classes (classes = [...])
  
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

#### Transformations appliquées :
- Rotation
- Flip horizontal
- Translation
- Échelle
- Changement HSV
- Perspective

### 2. 🔁 Conversion XML ➜ YOLO

python convert_voc_to_text.py



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
