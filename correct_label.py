import os

# Ton chemin de labels
label_dir = r"C:/Users/Utilisateur/Desktop/PublicationSpringer/IA/Dataset/validation/labels"

def clamp(value):
    return max(0.0, min(1.0, value))

def correct_labels(path):
    modified = False
    corrected_lines = []

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"⚠️ Ligne invalide ignorée dans {path} : {line.strip()}")
                continue
            cls, x, y, w, h = map(float, parts)
            x, y, w, h = map(clamp, [x, y, w, h])
            corrected_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            modified = True

    if modified:
        with open(path, "w") as f:
            f.writelines(corrected_lines)
        print(f"✅ Corrigé : {os.path.basename(path)}")

# Boucle sur tous les fichiers .txt
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        correct_labels(file_path)