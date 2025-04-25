import os
import xml.etree.ElementTree as ET
import csv

# Dossier contenant les fichiers d'annotations VOC
annotations_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\labels_VOC"
# Dossier parent des images (supposées avoir le même nom que les fichiers XML mais en .jpg ou .png)
images_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\images"
# Chemin de sortie du fichier CSV
output_csv = "./annotations_benchmark.csv"

# Fonction pour parser un fichier XML
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_filename = root.find("filename").text
    image_path = os.path.join(images_dir, image_filename)

    annotations = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        x1 = int(bndbox.find("xmin").text)
        y1 = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)
        annotations.append([image_path, x1, y1, x2, y2, class_name])
    return annotations

# Collecter toutes les annotations
all_annotations = []

for filename in os.listdir(annotations_dir):
    if filename.endswith(".xml"):
        xml_path = os.path.join(annotations_dir, filename)
        ann = parse_voc_annotation(xml_path)
        all_annotations.extend(ann)

# Écriture dans le fichier CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "x1", "y1", "x2", "y2", "class_id"])
    for row in all_annotations:
        writer.writerow(row)

print(f"Fichier {output_csv} généré avec succès.")