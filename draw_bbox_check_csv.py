import os
import cv2
import pandas as pd

# === CONFIGURATION ===
CSV_PATH = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\annotations.csv"
OUTPUT_DIR = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\test_annotation_crack_csv_origine"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CHARGEMENT DU CSV ===
df = pd.read_csv(CSV_PATH, header=None, sep=",")  # ou sep=",", selon ton fichier
df.columns = ["image_path", "xmin", "ymin", "xmax", "ymax", "class"]

# Nettoyage éventuel
df["image_path"] = df["image_path"].str.strip()

# === GROUPEMENT PAR IMAGE ===
for img_path, group in df.groupby("image_path"):
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Image introuvable : {img_path}")
        continue

    for _, row in group.iterrows():
        xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        label = str(row["class"])

        # Dessin du rectangle et du texte
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sauvegarde de l'image annotée
    filename = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, img)
    print(f"✅ Image annotée sauvegardée : {out_path}")