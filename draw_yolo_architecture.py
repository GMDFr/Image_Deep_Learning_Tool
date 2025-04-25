import yaml
yaml_path = r"C:\Users\Utilisateur\Desktop\PublicationSpringer\IA\yolov5\models\yolov5m.yaml"  # <--- ton fichier YAML ici

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch


# === CONFIGURATION COULEURS PAR MODULE ===
MODULE_COLORS = {
    'Conv': '#4C72B0',
    'C3': '#55A868',
    'SPPF': '#C44E52',
    'Concat': '#8172B3',
    'nn.Upsample': '#CCB974',
    'Detect': '#64B5CD',
    'default': '#8C8C8C'
}

def get_module_color(name):
    return MODULE_COLORS.get(name, MODULE_COLORS['default'])

# === CHARGEMENT DU FICHIER YAML ===
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

layers = config['backbone'] + config['head']

# === CONSTRUCTION DU GRAPHE ===
G = nx.DiGraph()
labels = {}
module_types = {}

for i, layer in enumerate(layers):
    from_idx, num, module, args = layer
    G.add_node(i)
    labels[i] = f"{i}: {module}"
    module_types[i] = module
    if isinstance(from_idx, list):
        for f in from_idx:
            G.add_edge(f, i)
    else:
        G.add_edge(from_idx, i)

# === DISPOSITION MANUELLE EN VERTICAL ===
# Chaque node sur un axe Y, espacés verticalement
layer_spacing = 2
node_height = 1.5
pos = {i: (0, -i * layer_spacing) for i in G.nodes}

# === PLOT EN RECTANGLES ===
fig, ax = plt.subplots(figsize=(8, 18))  # Portrait

# Dessin des blocs

for node, (x, y) in pos.items():
    if node == -1 or node not in module_types:
        continue  # ignore les faux noeuds
    width = 4
    height = node_height
    color = get_module_color(module_types.get(node, 'default'))


    rect = Rectangle((x - width / 2, y - height / 2), width, height,
                     facecolor=color, edgecolor='black', lw=1.2)
    ax.add_patch(rect)
    ax.text(x, y, labels[node], ha='center', va='center', fontsize=8, color='white')
    if isinstance(from_idx, list):
        for f in from_idx:
            if f != -1:
                G.add_edge(f, i)
    else:
        if from_idx != -1:
            G.add_edge(from_idx, i)
# Dessin des flèches
for u, v in G.edges:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    ax.annotate("",
                xy=(x2, y2 + node_height / 2),
                xytext=(x1, y1 - node_height / 2),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.2))

# Légende
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=mod)
                   for mod, color in MODULE_COLORS.items() if mod != 'default']
plt.legend(handles=legend_elements, loc='upper right', title="Modules", fontsize=8)

ax.set_xlim(-5, 5)
ax.set_ylim(-(len(G.nodes)+2) * layer_spacing, 2)
ax.set_aspect('equal')
ax.axis('off')

# === LÉGENDE DES COULEURS ===
legend_elements = [
    Patch(facecolor='#4daf4a', edgecolor='black', label='Conv'),
    Patch(facecolor='#377eb8', edgecolor='black', label='C3'),
    Patch(facecolor='#984ea3', edgecolor='black', label='SPPF'),
    Patch(facecolor='#ff7f00', edgecolor='black', label='Upsample'),
    Patch(facecolor='#e41a1c', edgecolor='black', label='Concat'),
    Patch(facecolor='#f781bf', edgecolor='black', label='Detect')
]

plt.legend(handles=legend_elements, title="Modules", loc='center left', bbox_to_anchor=(1.0, 0.5))


plt.tight_layout()
plt.savefig("yolov5_architecture_styled.png", dpi=300, bbox_inches='tight')
plt.show()