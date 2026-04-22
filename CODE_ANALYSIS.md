# Analyse du Projet Background-Segmentation

Ce projet est une suite de benchmarking complète pour les modèles de **Video Matting** (détourage vidéo / segmentation d'arrière-plan). Il permet d'évaluer la performance (latence, FLOPs) et la qualité (IoU, stabilité temporelle) de différents modèles de deep learning.

## 🏗️ Architecture du Projet

Le projet est structuré de manière modulaire autour du répertoire `benchmark/` :

```text
benchmark/
├── models/             # Implémentations des modèles (Mediapipe, RVM, MODNet, etc.)
├── runner.py           # Moteur principal (orchestration inférence + évaluation)
├── metrics.py          # Calcul des métriques de qualité
├── run_benchmark.py    # Point d'entrée CLI
├── dashboard.py        # Interface web Streamlit
├── config.py           # Paramètres globaux et chemins
└── requirements.txt    # Dépendances Python
```

---

## 🚀 Composants Principaux

### 1. Moteur de Benchmark (`runner.py`)
C'est le cœur du système. Il gère le cycle de vie d'un test :
- **Découverte** : Il cherche des couples (vidéo source, ground truth) dans les dossiers configurés.
- **Inférence** : Charge un modèle, traite la vidéo frame par frame, et mesure la latence (p95, moyenne) ainsi que les FLOPs.
- **Évaluation** : Compare les masques générés au "Ground Truth" (vérité terrain) pour calculer les scores de qualité.
- **Export** : Sauvegarde les résultats en CSV/JSON et peut générer des vidéos du sujet détouré.

### 2. Modèles (`models/`)
Le projet utilise une architecture de "Wrappers". Chaque modèle (MediaPipe, RVM, MODNet, EfficientViT, etc.) hérite d'une classe de base `BaseModelWrapper` (`base.py`). 
Cela permet :
- Une interface unifiée : `load()`, `predict(frame)`, `reset_state()`, `cleanup()`.
- Une facilité d'ajout de nouveaux modèles.

### 3. Métriques (`metrics.py`)
Le projet ne se contente pas de mesurer la vitesse, il évalue finement la qualité :
- **IoU (Intersection over Union)** : Mesure globale de la précision du masque.
- **Boundary F-measure** : Précision spécifique sur les contours (critique pour les cheveux/doigts).
- **Flow Warping Error (FWE)** : Mesure la stabilité temporelle. Elle utilise le flux optique pour vérifier que le masque ne "tremble" pas d'une image à l'autre.

### 4. Interfaces
- **CLI (`run_benchmark.py`)** : Pour lancer des benchmarks massifs via terminal avec des options comme `--models`, `--save-masks`, ou `--shuffle`.
- **Dashboard (`dashboard.py`)** : Une interface moderne sous **Streamlit** pour lancer des tests, visualiser les résultats en direct sous forme de tableaux et graphiques, et comparer les modèles visuellement.

---

## 🛠️ Fonctionnement Technique

1. **Chargement de la vidéo** : Utilise OpenCV pour extraire les frames.
2. **Inférence** : 
   - Le modèle traite chaque image.
   - Les latences sont enregistrées individuellement (en ignorant les premières frames de "warm-up").
   - Les masques sont stockés temporairement sur le disque pour ne pas saturer la RAM.
3. **Calcul des métriques** :
   - Les masques prédits sont rechargés et comparés aux masques de référence.
   - Le flux optique (Farneback) est utilisé pour le calcul de la stabilité.
4. **Nettoyage et Rapport** : Les fichiers temporaires sont supprimés et un rapport détaillé est généré dans `output/`.

---

## 📊 Métriques Mesurées

| Métrique | Description | Importance |
| :--- | :--- | :--- |
| **IoU** | Précision spatiale | Précision globale de la silhouette. |
| **Boundary F** | Précision des bords | Qualité des détails fins (cheveux). |
| **FWE** | Stabilité temporelle | Absence de scintillement (flickering). |
| **p95 Latency** | Performance réelle | Temps de traitement pour 95% des frames. |
| **FLOPs** | Charge computationnelle | Complexité théorique du modèle. |
