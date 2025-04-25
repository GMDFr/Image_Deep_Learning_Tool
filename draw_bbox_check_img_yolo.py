import os
import cv2
from pathlib import Path

# === CONFIGURATION ===
image_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\images")  # par exemple: Path("Dataset/images/train")
label_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\labels")  # par exemple: Path("Dataset/labels/train")
output_dir = Path(r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train_img_bbox_draw")  # dossier où sauvegarder les images annotées

output_dir.mkdir(parents=True, exist_ok=True)

# === CLASSES (à adapter selon ton dataset YAML) ===
class_names = ["crack", "crazing", "inclusion","patches","pitted_surface","rolled-in_scale", "scratches"]

# === DESSIN DES BBOX ===
for img_path in image_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    label_path = label_dir / (img_path.stem + ".txt")
    
    if not label_path.exists():
        print(f"Pas d'annotation pour {img_path.name}")
        continue
    
    with open(label_path, "r") as f:
        for line in f:
            cls_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
            
            # Convertir YOLO -> pixels
            x1 = int((x_center - bbox_w / 2) * w)
            y1 = int((y_center - bbox_h / 2) * h)
            x2 = int((x_center + bbox_w / 2) * w)
            y2 = int((y_center + bbox_h / 2) * h)
            
            # Classe
            label = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))

            # Dessiner la bbox
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Enregistrer l'image annotée
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)
    print(f"Image sauvegardée : {output_path}")

print("\n✅ Terminé : Les images annotées sont dans :", output_dir)