import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add as module
from create_yaml import create_data_yaml
    
'''
- Temps de calcul.
- Calcul du temps d'inférence.
- Images après training avec la bounding box.
- Métriques de classification / détection (Seaborn, ultralytics) avec grilles, axes en anglais.
- Sauvegarde des résultats.  
- Stocke les résultats dans un DataFrame et export csv.

Etat de l'Art
=> YOLO V8 : check pourquoi ne pas l'utiliser
=> Prévoir tableau pour l'Etat de l'art. 
Cas d’usage	Modèle recommandé	Raisons
Détection temps réel sur caméra	YOLOv5 ou YOLOv8	Rapide, facile à déployer
Détection sur images haute résolution (analyse fine)	Faster R-CNN	Meilleure précision, gère bien les petits objets
Projet embarqué/temps réel léger	SSD	Compromis vitesse/précision
Étude comparative (benchmark)	YOLO, SSD, Faster R-CNN	Pour analyser avantages/inconvénients
AVEC REFERENCE BIBLIO. 

'''
'''
def train_yolo(data_yaml, epochs=50, batch_size=16, weights='yolov5m.pt', save_dir='runs/train'):  
    model = YOLO(weights)
    results = model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, save_dir=save_dir)
    return results
'''
#def train_yolo(data_yaml, epochs=50, weights='yolov5mu.pt', save_dir='runs/train'):  
   # model = YOLO(weights)
   # results = model.train(data=data_yaml, epochs=epochs, save_dir=save_dir)
   # return results

'''
def train_yolo_quick_test(data_yaml, weights='yolov5mu.pt', save_dir='\IA\Dataset\RESULTS\runs\train_test'):
    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=320,              # image size réduite pour entraînement rapide
        batch=4,                # petit batch size pour tests rapides
        save_dir=save_dir,
        verbose=True,          # désactive les logs détaillés
        workers=0,              # évite les threads multiples (plus stable en test rapide)
        device=0                # utilise le GPU si dispo (sinon, adapter)
    )
    return results
'''
def train_yolo_quick_test(data_yaml, weights='yolov5mu.pt', save_dir='\IA\Dataset\RESULTS\runs\final_train_yolo'):
    '''
    Nb d'images : 11754

    '''
    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=320,              # image size réduite pour entraînement rapide
        batch=32,                # petit batch size pour tests rapides
        save_dir=save_dir,
        verbose=True,          # désactive les logs détaillés
        workers=0,              # évite les threads multiples (plus stable en test rapide)
        device=0                # utilise le GPU si dispo (sinon, adapter)
    )
    return results

##################
def train_ssd(data_yaml, epochs=50, batch_size=16, weights='yolov5m.pt', save_dir='runs/train'):  
    model = YOLO(weights)
    results = model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, save_dir=save_dir)
    return results

def train_faster_rcnn(data_yaml, epochs=50, batch_size=16, weights='yolov5m.pt', save_dir='runs/train'):  
    model = YOLO(weights)
    results = model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, save_dir=save_dir)
    return results
##################


def plot_training_results(results):
    metrics = ['train/box_loss', 'train/cls_loss', 'train/obj_loss', 'val/box_loss', 'val/cls_loss', 'val/obj_loss']
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        plt.plot(results.metrics[metric], label=metric)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.show()

def evaluate_model(model, data_yaml, save_dir='runs/val'):
    results = model.val(data=data_yaml, save_dir=save_dir)
    print(f"Results saved in {save_dir}")
    return results

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, save_path='best_yolo_model.pt'):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Configuration YAML path
data_yaml_path = r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\data.yaml'
create_data_yaml(data_yaml_path)

# Train YOLO model
#results = train_yolo(data_yaml_path)
#plot_training_results(results)

# QUICK TEST YOLO MODEL
data_yaml_path = r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\data.yaml'
results = train_yolo_quick_test(data_yaml_path)

# Load trained model and evaluate
yolo_model = YOLO(results.best_model_path)
eval_results = evaluate_model(yolo_model, data_yaml_path)

# Extract predictions and ground truths for confusion matrix
class_names = ['crack','crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
y_true = eval_results.metrics['val/cls']  # Adjust based on actual output
y_pred = eval_results.metrics['pred/cls']  # Adjust based on actual output
plot_confusion_matrix(y_true, y_pred, class_names)

# Save model
save_model(yolo_model, r'C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\Dataset\model\yolo_best_model.pt')
