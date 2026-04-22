# Audit de Performance : Benchmark Video Matting

Cet audit détaille les goulots d'étranglement actuels du flux de benchmark et propose des solutions techniques concrètes pour accélérer le traitement.

---

## 1. Chargement et Décodage des Frames
**État actuel :** Utilsation de `cv2.VideoCapture` et stockage de **toutes** les frames dans une liste Python (`List[np.ndarray]`).
- 🛑 **Goulot :** Saturation de la RAM sur les vidéos longues. Une vidéo 4K de 10s peut consommer plusieurs Go.
- 🛑 **Goulot :** `cv2.read()` est une opération bloquante et mono-threadée.

**Améliorations proposées :**
- **Librairie `decord`** : Utiliser `decord.VideoReader` qui utilise des wrappers matériels plus optimisés et permet un accès "Lazy" aux frames.
- **Pattern Iterator** : Transformer `_read_video_frames` en générateur. Cela permet de traiter des vidéos de n'importe quelle longueur avec une consommation RAM constante.
- **Async Loader** : Implémenter un `Queue` (Producteur/Consommateur) où un thread décode les frames pendant que le modèle les traite.

---

## 2. Inférence (Appel des Modèles)
**État actuel :** Boucle séquentielle `for frame in frames: model.predict(frame)`.
- 🛑 **Goulot :** Temps d'attente CPU entre deux inférences. Le NPU (Neural Engine) des Mac Silicon n'est pas saturé.
- 🛑 **Goulot :** Les modèles ONNX sont souvent appelés avec `CPUExecutionProvider` par défaut si non configurés.

**Améliorations proposées :**
- **Inférence par Batch** : Des modèles comme **MODNet** supportent déjà le batching. Passer plusieurs frames à la fois (ex: batch de 8) permet d'optimiser l'usage des registres SIMD et du cache L3 du processeur.
- **Accélération Mac (CoreML)** : S'assurer que `CoreMLExecutionProvider` est actif. Conversion des modèles en **FP16** pour diviser la bande passante mémoire par deux sans perte de qualité notable.
- **Quantization** : Utiliser des modèles quantifiés en `INT8` (pour MediaPipe/TFLite) si la précision le permet.

---

## 3. Sauvegarde des Vidéos et Masques
**État actuel :** Compilation via `imageio` et `cv2.imwrite`.
- 🛑 **Goulot :** L'encodage H.264 par soft (CPU) est extrêmement gourmand.

**Améliorations proposées :**
- **Hardware Acceleration (VideoToolbox)** : Sur macOS, utiliser obligatoirement l'encodeur matériel :
  ```python
  writer = imageio.get_writer(..., codec='h264_videotoolbox', bitrate='5M')
  ```
- **Parallel Writing** : Utiliser `concurrent.futures.ThreadPoolExecutor` pour les sauvegardes de masques individuels.
- **FFmpeg Pipes** : Utiliser la bibliothèque `ffmpeg-python` pour piper directement les frames NumPy vers le processus FFmpeg sans passer par le disque.

---

## 4. Calcul des Métriques
**État actuel :** Calcul séquentiel sur un seul cœur CPU.
- 🛑 **Goulot :** Le calcul du flux optique Farneback est l'opération la plus lente du benchmark.

**Améliorations proposées :**
- **DIS Optical Flow** : Remplacer `Farneback` par `cv2.optflow.createOptFlow_DIS()`. C'est l'algorithme "State-of-the-art" pour la vitesse dans OpenCV, souvent 10x plus rapide que Farneback.
- **Multiprocessing** : Utiliser `ProcessPoolExecutor` (pas `ThreadPool` à cause du GIL global de Python) pour calculer les scores IoU et Boundary F sur tous les cœurs du Mac (M1/M2/M3 ont 8+ cœurs).
- **GPU Metrics** : Calculer l'IoU en utilisant `PyTorch` ou `CuPy` si disponible, permettant de traiter 1000 frames en quelques millisecondes.

---

## 🛠️ Plan d'Action Technique

1.  **Refactoring Logic** : Remplacer `_read_video_frames` par un itérateur de batch.
2.  **Changement d'Algo** : Passer au `DISFlow` pour la métrique de stabilité temporelle (FWE).
3.  **Encodage Matériel** : Configurer la compilation vidéo pour utiliser `videotoolbox`.
4.  **Parallélisation** : Distribuer l'évaluation sur tous les cœurs CPU via `multiprocessing`.

---

## Comparatif des Gains Potentiels (Estimés)
- **Décodage** : +30% (Decord)
- **Inférence** : +2x à +5x (Batching + CoreML)
- **Sauvegarde** : +4x (VideoToolbox)
- **Métriques** : +8x (Multiprocessing sur un Mac 8-coeurs)
