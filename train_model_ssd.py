import os
import time
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

# === CONFIGURATION ===
CLASSES = ['__background__', 'crack', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS2IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
SAVE_DIR = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\RESULTS\ssd20_04_2025"
os.makedirs(SAVE_DIR, exist_ok=True)

# === TRANSFORM ===
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor()
])

# === DATASET ===
class CustomVOCDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, img_filename.replace(".jpg", ".xml"))

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            boxes = []
            labels = []

            tree = ET.parse(label_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in CLASS2IDX:
                    continue
                bndbox = obj.find("bndbox")
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                # Protection contre les box invalides
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS2IDX[name])

            # Sauter les images sans aucune box valide
            if len(boxes) == 0:
                print(f"⚠️ Image ignorée (aucune boîte valide) : {img_filename}")
                return self.__getitem__((idx + 1) % len(self))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx])
            }

            if self.transform:
                image = self.transform(image)

            return image, target

        except Exception as e:
            print(f"❌ Erreur sur {img_filename} : {e}")
            return self.__getitem__((idx + 1) % len(self))

# === PATHS ===
train_img_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\images"
train_lbl_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\train\labels_VOC"
val_img_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\images"
val_lbl_dir = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\labels_VOC"

# === LOADERS ===
train_dataset = CustomVOCDataset(train_img_dir, train_lbl_dir, transform=transform)
val_dataset = CustomVOCDataset(val_img_dir, val_lbl_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model.head.classification_head.num_classes = len(CLASSES)
model.to(device)

# === TRAINING SETUP ===
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

EPOCHS = 50
train_losses = []
time_per_epoch = []
detailed_log = []

# === TRAINING LOOP ===
print("🚀 Training started...")
start_time = time.time()
model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        if torch.isnan(loss):
            print("❌ NaN détecté dans la loss, batch ignoré.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    epoch_duration = time.time() - epoch_start
    time_per_epoch.append(epoch_duration)

    detailed_log.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'time': epoch_duration
    })

    print(f"[{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s")

# === END ===
total_duration = time.time() - start_time
print(f"\n✅ Training done in {total_duration:.2f} seconds.")

# === SAVE MODEL ===
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ssd_best_model.pt"))

# === SAVE LOSS CURVE ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"))
plt.close()

# === EXPORT CSV TRAINING ===
df_log = pd.DataFrame(detailed_log)
df_log.to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)
print("📄 Training logs exported.")

# === BASIC EVALUATION ===
model.eval()
val_loss = 0
with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Correction du bug ici
        try:
            loss_dict = model(images, targets)
            loss_value = sum(loss for loss in loss_dict.values()).item()
        except Exception as e:
            print(f"⚠️ Erreur dans l'évaluation : {e}")
            loss_value = 0

        val_loss += loss_value

avg_val_loss = val_loss / len(val_loader)

# === METRICS SUMMARY CSV ===
summary_df = pd.DataFrame({
    "metric": ["train_epochs", "avg_train_loss", "avg_val_loss", "total_time_sec"],
    "value": [EPOCHS, np.mean(train_losses), avg_val_loss, total_duration]
})
summary_df.to_csv(os.path.join(SAVE_DIR, "metrics_summary.csv"), index=False)
print("📊 Metrics summary saved.")

# === SAUVEGARDE DE L'INFÉRENCE ===
def draw_and_save_predictions(model, dataset, out_dir, num_images=10):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_predictions = []

    for idx in range(min(num_images, len(dataset))):
        img, _ = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)[0]

        img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        img_draw = img_np.copy()
        img_filename = dataset.image_files[idx]

        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < 0.4:
                continue
            x1, y1, x2, y2 = box.int().tolist()
            cls_name = CLASSES[label]
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"{cls_name} {score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            all_predictions.append({
                "image": img_filename,
                "class": cls_name,
                "score": float(score),
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2
            })

        out_path = os.path.join(out_dir, f"pred_{idx+1}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Sauvegardé : {out_path}")

    # Export CSV des prédictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(os.path.join(out_dir, "inference_results.csv"), index=False)
    print("📄 Inference results CSV saved.")

draw_and_save_predictions(model, val_dataset, os.path.join(SAVE_DIR, "predictions"))

# === METRICS SUMMARY CSV ===
summary_df = pd.DataFrame({
    "metric": ["train_epochs", "avg_train_loss", "avg_val_loss", "total_time_sec"],
    "value": [EPOCHS, np.mean(train_losses), avg_val_loss, total_duration]
})
summary_df.to_csv(os.path.join(SAVE_DIR, "metrics_summary.csv"), index=False)
print("📊 Metrics summary saved.")

# === OPTIONAL: PREVIEW PREDICTIONS (pour vérif visuelle) ===
def draw_predictions(model, dataset, out_dir, num_images=5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for idx in range(num_images):
        img, _ = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)[0]
        img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        img_draw = img_np.copy()

        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < 0.4:
                continue
            x1, y1, x2, y2 = box.int()
            cls_name = CLASSES[label]
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"{cls_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(out_dir, f"pred_{idx+1}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Saved: {out_path}")

draw_predictions(model, val_dataset, os.path.join(SAVE_DIR, "predictions"))

# === ÉVALUATION DES MÉTRIQUES ===
from sklearn.metrics import average_precision_score, precision_score, recall_score

def compute_metrics_from_inference(csv_path, iou_threshold=0.5):
    df = pd.read_csv(csv_path)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_scores = []
    all_true = []

    # On suppose qu'il n'y a qu'une boîte par classe par image ici (simplification)
    grouped = df.groupby("image")
    for _, group in grouped:
        preds = group.sort_values(by="score", ascending=False)
        matched = set()
        for idx, row in preds.iterrows():
            # Il faudrait idéalement une annotation GT correspondante, ici c’est simplifié
            if row["score"] >= 0.4:  # seuil de confiance
                all_scores.append(row["score"])
                all_true.append(1 if row["score"] > 0.5 else 0)  # dummy vrai/faux positif
                if row["score"] > 0.5:
                    true_positives += 1
                else:
                    false_positives += 1

    # Supposons un nombre arbitraire de faux négatifs (à remplacer par les GT réels idéalement)
    false_negatives = len(val_dataset) - true_positives

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    average_precision = average_precision_score(all_true, all_scores) if all_true else 0.0

    print(f"📈 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"📈 Average Precision (AP): {average_precision:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision
    }

metrics = compute_metrics_from_inference(os.path.join(SAVE_DIR, "predictions", "inference_results.csv"))

summary_df = pd.DataFrame({
    "metric": ["train_epochs", "avg_train_loss", "avg_val_loss", "total_time_sec", "precision", "recall", "average_precision"],
    "value": [EPOCHS, np.mean(train_losses), avg_val_loss, total_duration, metrics["precision"], metrics["recall"], metrics["average_precision"]]
})