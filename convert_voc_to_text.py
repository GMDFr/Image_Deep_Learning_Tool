import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def convert_voc_to_yolo(xml_folder, output_folder, classes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    print(f"[INFO] Classe '{cls}' ignorée dans {xml_file}")
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (
                    int(xmlbox.find('xmin').text),
                    int(xmlbox.find('ymin').text),
                    int(xmlbox.find('xmax').text),
                    int(xmlbox.find('ymax').text)
                )
                bb = convert((w, h), b)
                out_file.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")

        print(f"[OK] Converti : {xml_file} -> {txt_filename}")

# === CHEMINS D'ENTRÉE / SORTIE ===
train_xml_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\labels"
val_xml_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\labels"

train_output_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\labels_yolo"
val_output_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\labels_yolo"

# === CLASSES ===
classes = ['crack', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# === CONVERSION ===
print("==> Conversion des annotations TRAIN")
convert_voc_to_yolo(train_xml_dir, train_output_dir, classes)

print("==> Conversion des annotations VALIDATION")
convert_voc_to_yolo(val_xml_dir, val_output_dir, classes)