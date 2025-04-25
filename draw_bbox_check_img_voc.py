import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
image_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\images")
label_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\ARCHIVES\labels_VOC")
output_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train_img_bbox_draw_VOC")

output_dir.mkdir(parents=True, exist_ok=True)

# === CLASSES (si besoin pour affichage) ===
# Tu peux laisser vide ou adapter selon ton jeu de données
# Si les classes sont déjà dans les fichiers XML, pas besoin de les définir ici
class_names = ["crack", "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# === DESSIN DES BBOX ===
for xml_file in label_dir.glob("*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Récupérer le nom de l'image
    filename = root.find("filename").text
    img_path = image_dir / filename
    if not img_path.exists():
        print(f"Image non trouvée : {filename}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Erreur de lecture image : {img_path}")
        continue

    # Lire la taille de l'image
    h, w = img.shape[:2]

    for obj in root.findall("object"):
        label = obj.find("name").text

        bbox = obj.find("bndbox")
        x1 = int(float(bbox.find("xmin").text))
        y1 = int(float(bbox.find("ymin").text))
        x2 = int(float(bbox.find("xmax").text))
        y2 = int(float(bbox.find("ymax").text))

        # Dessiner la bbox
        color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = output_dir / filename
    cv2.imwrite(str(output_path), img)
    print(f"✅ Image sauvegardée : {output_path}")

print("\n🎉 Terminé : Les images avec bbox VOC sont dans :", output_dir)