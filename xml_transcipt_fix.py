import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom

# === CONFIGURATION ===
CSV_PATH = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\annotations.csv"
XML_OUTPUT_DIR = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\crack_xml_fix"
os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

# === FONCTION POUR CREER UN XML PASCAL VOC ===
def create_voc_xml(image_path, image_shape, objects, output_path):
    height, width, depth = image_shape

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:
        obj_tag = ET.SubElement(annotation, "object")
        ET.SubElement(obj_tag, "name").text = obj["class"]
        ET.SubElement(obj_tag, "pose").text = "Unspecified"
        ET.SubElement(obj_tag, "truncated").text = "0"
        ET.SubElement(obj_tag, "difficult").text = "0"

        bbox = ET.SubElement(obj_tag, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bbox, "ymax").text = str(obj["ymax"])

    # Beautify and write
    xml_str = ET.tostring(annotation)
    parsed = minidom.parseString(xml_str)
    pretty_xml_as_str = parsed.toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml_as_str)

# === CHARGEMENT DU CSV ===
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["image_path", "xmin", "ymin", "xmax", "ymax", "class"]

# === TRAITEMENT ===

for img_path, group in df.groupby("image_path"):

    if not os.path.isfile(img_path):
        print(f"❌ Chemin invalide dans le CSV : {img_path}")
        continue

    img_path = os.path.normpath(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Image introuvable : {img_path}")
        continue
    h, w, c = img.shape
    objects = group[["xmin", "ymin", "xmax", "ymax", "class"]].to_dict("records")
    xml_filename = os.path.splitext(os.path.basename(img_path))[0] + ".xml"
    xml_path = os.path.join(XML_OUTPUT_DIR, xml_filename)
    create_voc_xml(img_path, (h, w, c), objects, xml_path)
    print(f"✅ Fichier XML créé : {xml_path}")