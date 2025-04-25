import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid conflict lib with OpenCV and Torch

import time
import torch
import torchvision.transforms as T
import torchvision
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, average_precision_score
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import subprocess
import threading
import csv
from torchvision.models.detection.ssd import SSDClassificationHead


# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_PATH = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\train_YOLO_M\weights\best.pt"
SSD_PATH = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\RESULTS\ssd20_04_2025\ssd_best_model.pt"
IMAGE_DIR = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\validation\images"
ANNOTATIONS = "./annotations_benchmark.csv"  # Nouveau format CSV

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# === Load models ===
yolo_model = yolo_model = YOLO(YOLO_PATH)
ssd_model = torchvision.models.detection.ssd300_vgg16(num_classes=7).to(DEVICE)
ssd_model.load_state_dict(torch.load(SSD_PATH, map_location=DEVICE))
ssd_model.eval()

# Charger SSD pré-entraîné
ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=False)
in_channels = ssd_model.head.classification_head.num_classes  # normalement 91
num_anchors = ssd_model.head.classification_head.num_anchors
ssd_model.head.classification_head = SSDClassificationHead(
    in_channels=ssd_model.head.classification_head.in_channels,
    num_anchors=num_anchors,
    num_classes=7
)


# === Load annotations from CSV
def load_annotations_csv(path):
    gt_boxes = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path']
            box = [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']), int(row['class_id'])]
            gt_boxes.setdefault(image_path, []).append(box)
    return gt_boxes

ground_truth = load_annotations_csv(ANNOTATIONS)

# === Ground truth fetcher
def get_boxes_for_image(image_name, width, height):
    img_path = os.path.join(IMAGE_DIR, image_name + ".jpg")
    return ground_truth.get(img_path, [])

# === Energy monitor 
class PowerMonitor:
    def __init__(self):
        self.running = False
        self.data = []

    def _run(self):
        process = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while self.running:
            line = process.stdout.readline().decode("utf-8")
            if "VDD_CPU_GPU_CV" in line:
                try:
                    part = line.split("VDD_CPU_GPU_CV:")[1].split(" ")[1]
                    power = int(part)
                    self.data.append(power)
                except Exception:
                    continue
        process.terminate()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_average_power(self):
        return round(np.mean(self.data), 2) if self.data else 0.0

# === Compute IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# === Courbes
def plot_precision_recall(y_true, y_scores, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.2f})', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = f"{model_name.replace(' ', '_')}_PR_curve.png"
    plt.savefig(path)
    plt.close()

def plot_f1_vs_threshold(y_true, y_scores, model_name):
    thresholds = np.linspace(0.0, 1.0, 100)
    f1_scores = [f1_score(y_true, (np.array(y_scores) > t).astype(int)) for t in thresholds]
    plt.figure()
    plt.plot(thresholds, f1_scores, color='green')
    plt.xlabel('Seuil de Confiance')
    plt.ylabel('F1-score')
    plt.title(f'F1-score vs. Seuil - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    path = f"{model_name.replace(' ', '_')}_F1_vs_Threshold.png"
    plt.savefig(path)
    plt.close()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='red', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    path = f"{model_name.replace(' ', '_')}_ROC_curve.png"
    plt.savefig(path)
    plt.close()

# === Résultats LaTeX
latex_results = []
def store_latex_result(model, precision, recall, ap, iou, time_taken, power):
    latex_results.append({
        "Model": model,
        "Precision": precision,
        "Recall": recall,
        "mAP": ap,
        "IoU": iou,
        "Time": time_taken,
        "Power": power
    })

def generate_latex_comparison_table():
    header = """
\\begin{table}[h!]
\\centering
\\caption{Comparaison des performances et de la consommation énergétique des modèles YOLOv5 et SSD300 sur Jetson Xavier NX.}
\\begin{tabular}{|l|c|c|}
\\hline
\\textbf{Metric} & \\textbf{YOLOv5m} & \\textbf{SSD300} \\\\
\\hline
"""
    footer = """
\\hline
\\end{tabular}
\\label{tab:comparison_yolo_ssd}
\\end{table}
"""
    yolo = next((r for r in latex_results if "YOLO" in r["Model"]), None)
    ssd = next((r for r in latex_results if "SSD" in r["Model"]), None)
    rows = f"""Precision & {yolo['Precision']:.3f} & {ssd['Precision']:.3f} \\\\
Recall & {yolo['Recall']:.3f} & {ssd['Recall']:.3f} \\\\
mAP & {yolo['mAP']:.3f} & {ssd['mAP']:.3f} \\\\
Mean IoU & {yolo['IoU']:.3f} & {ssd['IoU']:.3f} \\\\
Execution Time (s) & {yolo['Time']:.2f} & {ssd['Time']:.2f} \\\\
Average Power (mW) & {yolo['Power']:.2f} & {ssd['Power']:.2f} \\\\
"""
    table = header + rows + footer
    with open("comparison_yolo_ssd.tex", "w") as f:
        f.write(table)

# === Inference Wrappers
def infer_yolo(img):
    # Convertir image PIL en numpy
    img_np = np.array(img)
    results = yolo_model.predict(img_np, imgsz=640, conf=CONF_THRESHOLD, device=0 if DEVICE.type == "cuda" else "cpu")

    preds = results[0].boxes  # results[0] correspond à une seule image
    boxes = preds.xyxy.cpu().numpy() if preds.xyxy is not None else []
    scores = preds.conf.cpu().numpy() if preds.conf is not None else []
    
    return np.hstack([boxes[:, :4], scores[:, None]]) if len(boxes) > 0 else []

def infer_ssd(img):
    img_tensor = ssd_transform(img).to(DEVICE)
    with torch.no_grad():
        output = ssd_model([img_tensor])[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    return np.hstack([boxes, scores[:, None]])

# === Evaluation function
def evaluate(model_name, model_func):
    power_monitor = PowerMonitor()
    power_monitor.start()

    start_time = time.time()
    y_true, y_pred, y_score, ious = [], [], [], []
    image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        labels = get_boxes_for_image(img_path.stem, width, height)

        detections = model_func(img)
        matched = [False] * len(labels)

        for det in detections:
            if det[4] < CONF_THRESHOLD:
                continue
            pred_box = list(map(int, det[:4]))
            score = det[4]
            found_match = False
            for i, gt_box in enumerate(labels):
                iou = compute_iou(pred_box, gt_box[:4])
                if iou >= IOU_THRESHOLD and not matched[i]:
                    matched[i] = True
                    ious.append(iou)
                    y_true.append(1)
                    y_pred.append(1)
                    y_score.append(score)
                    found_match = True
                    break
            if not found_match:
                y_true.append(0)
                y_pred.append(1)
                y_score.append(score)

        for m in matched:
            if not m:
                y_true.append(1)
                y_pred.append(0)
                y_score.append(0.0)

    power_monitor.stop()
    end_time = time.time()
    avg_power = power_monitor.get_average_power()
    total_time = end_time - start_time

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_score)
    mean_iou = np.mean(ious) if ious else 0

    plot_precision_recall(y_true, y_score, model_name)
    plot_f1_vs_threshold(y_true, y_score, model_name)
    plot_roc_curve(y_true, y_score, model_name)

    store_latex_result(model_name, precision, recall, ap, mean_iou, total_time, avg_power)

    print(f"\n📊 Results for {model_name}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"mAP:       {ap:.3f}")
    print(f"Mean IoU:  {mean_iou:.3f}")

# === Main
if __name__ == "__main__":
    evaluate("YOLOv5m", infer_yolo)
    evaluate("SSD300-VGG16", infer_ssd)
    generate_latex_comparison_table()