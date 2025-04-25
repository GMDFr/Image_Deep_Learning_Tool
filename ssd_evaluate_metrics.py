from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os

all_preds = []
all_targets = []
all_scores = []

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_labels = output['labels'].cpu().numpy()
            true_labels = target['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()

            all_preds.extend(pred_labels)
            all_targets.extend(true_labels)
            all_scores.extend(scores[:len(pred_labels)])  # Aligné

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_scores = np.array(all_scores)


def save_plot(path, fig):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# === Confusion Matrix brute ===
cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
save_plot(os.path.join(SAVE_DIR, "confusion_matrix.png"), fig)

# === Confusion Matrix normalisée ===
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Normalized Confusion Matrix")
save_plot(os.path.join(SAVE_DIR, "confusion_matrix_normalized.png"), fig)

# === F1-Score par classe ===
f1s = f1_score(all_targets, all_preds, labels=list(range(len(CLASSES))), average=None)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=CLASSES, y=f1s, ax=ax)
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score per Class")
ax.set_xticklabels(CLASSES, rotation=45)
ax.grid()
save_plot(os.path.join(SAVE_DIR, "f1_curve.png"), fig)

# === Label Count ===
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(all_targets, bins=len(CLASSES), kde=False)
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=45)
ax.set_title("Label Frequency")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
save_plot(os.path.join(SAVE_DIR, "labels.png"), fig)

# === Label Correlogram ===
label_df = pd.DataFrame({"True": all_targets, "Pred": all_preds})
fig, ax = plt.subplots(figsize=(8, 6))
corr = pd.crosstab(label_df["True"], label_df["Pred"], normalize='index')
sns.heatmap(corr, annot=True, cmap="coolwarm", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title("Label Correlogram")
save_plot(os.path.join(SAVE_DIR, "labels_correlogram.png"), fig)

# === Precision-Recall Curve (macro) ===
binarized_targets = label_binarize(all_targets, classes=range(len(CLASSES)))
binarized_preds = label_binarize(all_preds, classes=range(len(CLASSES)))

fig, ax = plt.subplots(figsize=(10, 7))
for i, class_name in enumerate(CLASSES):
    if binarized_targets[:, i].sum() == 0:
        continue
    precision, recall, _ = precision_recall_curve(binarized_targets[:, i], binarized_preds[:, i])
    ap = average_precision_score(binarized_targets[:, i], binarized_preds[:, i])
    ax.plot(recall, precision, label=f"{class_name} (AP={ap:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR Curve per Class")
ax.legend()
ax.grid()
save_plot(os.path.join(SAVE_DIR, "pr_curve.png"), fig)

# === R Curve (Recall only) ===
fig, ax = plt.subplots(figsize=(8, 5))
recall_scores = []
for i in range(len(CLASSES)):
    true = binarized_targets[:, i]
    pred = binarized_preds[:, i]
    recall = np.sum((true == 1) & (pred == 1)) / (np.sum(true) + 1e-6)
    recall_scores.append(recall)

sns.barplot(x=CLASSES, y=recall_scores, ax=ax)
ax.set_ylabel("Recall")
ax.set_title("Recall Curve per Class")
ax.set_xticklabels(CLASSES, rotation=45)
ax.grid()
save_plot(os.path.join(SAVE_DIR, "recall_curve.png"), fig)

# === P Curve (Precision only) ===
fig, ax = plt.subplots(figsize=(8, 5))
precision_scores = []
for i in range(len(CLASSES)):
    true = binarized_targets[:, i]
    pred = binarized_preds[:, i]
    precision = np.sum((true == 1) & (pred == 1)) / (np.sum(pred) + 1e-6)
    precision_scores.append(precision)

sns.barplot(x=CLASSES, y=precision_scores, ax=ax)
ax.set_ylabel("Precision")
ax.set_title("Precision Curve per Class")
ax.set_xticklabels(CLASSES, rotation=45)
ax.grid()
save_plot(os.path.join(SAVE_DIR, "precision_curve.png"), fig)

print("📊 Toutes les métriques avancées ont été générées.")