import os
import pandas as pd
import xml.etree.ElementTree as ET

# Chargement du fichier CSV avec entête
input_file = "annotations.csv"  # ou .txt selon le format
output_dir = "annotations_xml"
os.makedirs(output_dir, exist_ok=True)

# Lire avec Pandas (séparateur tabulation)
df = pd.read_csv(input_file, sep=",")

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

for _, row in df.iterrows():
    try:
        image_path = str(row["image_path"])
        x = int(float(row["x"]))
        y = int(float(row["y"]))
        w = int(float(row["w"]))
        h = int(float(row["h"]))
        label = str(row["label"])

        xmin = str(x)
        ymin = str(y)
        xmax = str(x + w)
        ymax = str(y + h)

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]

        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = "cr"

        fname = ET.SubElement(annotation, "filename")
        fname.text = filename

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "SDNET2028"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = "256"  # Adapter si besoin
        height = ET.SubElement(size, "height")
        height.text = "256"
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = label
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = xmin
        ET.SubElement(bndbox, "ymin").text = ymin
        ET.SubElement(bndbox, "xmax").text = xmax
        ET.SubElement(bndbox, "ymax").text = ymax

        xml_filename = base_name + ".xml"
        xml_path = os.path.join(output_dir, xml_filename)
        tree = ET.ElementTree(annotation)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Fichier XML écrit : {xml_path}")

    except Exception as e:
        print(f"Erreur sur la ligne {row.to_dict()}: {e}")
        continue