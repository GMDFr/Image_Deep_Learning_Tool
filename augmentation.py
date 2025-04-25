import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Clip les valeurs au cas où elles dépassent l'image
        original = (xmin, ymin, xmax, ymax)
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width - 1))
        ymax = max(0, min(ymax, height - 1))

        if original != (xmin, ymin, xmax, ymax):
            print(f"⚠️ BBox corrigée pour {filename}: {original} ➔ {(xmin, ymin, xmax, ymax)}")

        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    
    return filename, width, height, objects

def update_xml(xml_path, new_filename, objects, output_xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.find('filename').text = new_filename
    
    for obj, new_bbox in zip(root.findall('object'), objects):
        obj.find('bndbox/xmin').text = str(int(new_bbox['bbox'][0]))
        obj.find('bndbox/ymin').text = str(int(new_bbox['bbox'][1]))
        obj.find('bndbox/xmax').text = str(int(new_bbox['bbox'][2]))
        obj.find('bndbox/ymax').text = str(int(new_bbox['bbox'][3]))
    
    tree.write(output_xml_path)

def apply_and_save(transform_name, transform, img, bboxes, labels, base_filename, xml_path, output_dir_img, output_dir_xml):
    try:
        assert img is not None, f"Image {base_filename} could not be loaded!"
        assert len(bboxes) > 0, f"No bounding boxes found for {base_filename}"

        augmented = transform(image=img, 
                              bboxes=bboxes, 
                              class_labels=labels, 
                              clip_boxes = True)
        
        new_img = augmented['image']
        new_bboxes = augmented['bboxes']

        name_base = os.path.splitext(base_filename)[0]
        new_filename = f"{name_base}_{transform_name}.jpg"
        new_xml_name = f"{name_base}_{transform_name}.xml"
        
        new_img_path = os.path.join(output_dir_img, new_filename)
        new_xml_path = os.path.join(output_dir_xml, new_xml_name)

        if len(new_bboxes) == 0:
            print(f"⚠️ [WARNING] No bboxes after {transform_name} on {base_filename}. Saving image anyway (no annotation).")
            cv2.imwrite(new_img_path, new_img)
            return

        new_objects = [{'name': lbl, 'bbox': bbox} for lbl, bbox in zip(labels, new_bboxes)]
        cv2.imwrite(new_img_path, new_img)
        update_xml(xml_path, new_filename, new_objects, new_xml_path)

        print(f"✅ Saved {new_filename} + {new_xml_name}")

    except AssertionError as ae:
        print(f"❌ AssertionError for {base_filename}: {ae}")
    except Exception as e:
        print(f"❌ Error during {transform_name} on {base_filename}: {e}")

def augment_dataset(input_img_dir, input_xml_dir, output_img_dir, output_xml_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    print("📁 Démarrage de l'augmentation de la base d'images...\n")

    common_params = {
        'bbox_params': A.BboxParams(format='pascal_voc',
                                    label_fields=['class_labels']
                                    )
    }

    transformations = {
        "hsv": A.Compose([A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=70, val_shift_limit=40, p=1.0)], **common_params),
        "rotate": A.Compose([A.Rotate(limit=10, p=1.0)], **common_params),
        "translate": A.Compose([A.Affine(translate_percent=0.1, scale=1.0, rotate=0, shear=0, p=1.0)], **common_params),
        "scale": A.Compose([A.Affine(scale=(1.0, 1.1), translate_percent=0, rotate=0, shear=0, p=1.0)], **common_params),
        "perspective": A.Compose([A.Perspective(scale=(0.0001, 0.0005), p=1.0)], **common_params),
        "flip": A.Compose([A.HorizontalFlip(p=1.0)], **common_params)
    }

    for filename in os.listdir(input_img_dir):
        if filename.lower().endswith(".jpg"):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_img_dir, filename)
            xml_path = os.path.join(input_xml_dir, base_name + ".xml")

            if not os.path.exists(xml_path):
                print(f"⚠️ Annotation non trouvée pour {filename}, ignorée.")
                continue

            print(f"\n🔄 Traitement de : {filename}")

            filename_xml, width, height, objects = parse_xml(xml_path)
            img = cv2.imread(img_path)

            bboxes = [obj['bbox'] for obj in objects]
            labels = [obj['name'] for obj in objects]

            for t_name, transform in transformations.items():
                print(f"  ➤ Application transformation: {t_name}")
                apply_and_save(t_name, transform, img, bboxes, labels, filename, xml_path, output_img_dir, output_xml_dir)

    print("\n✅ Augmentation terminée pour tout le dataset.")

# Exécution
# Liste pour boucler en automatique 
# defect = ['crack','crazing']
# for i in defect : 
# f'({i})
augment_dataset(
    input_img_dir=r'C:\Users\GuillaumeMORIN-DUPON\Documents\PublicationSpringer\IA\Dataset\train\images\crack',
    input_xml_dir=r'C:\Users\GuillaumeMORIN-DUPON\Documents\PublicationSpringer\IA\Dataset\train\annotations',
    output_img_dir=r'C:\Users\GuillaumeMORIN-DUPON\Documents\PublicationSpringer\IA\Dataset\train_augmented_crack\images',
    output_xml_dir=r'C:\Users\GuillaumeMORIN-DUPON\Documents\PublicationSpringer\IA\Dataset\train_augmented_crack\annotations'
)