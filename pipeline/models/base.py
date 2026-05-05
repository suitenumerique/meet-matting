"""
Classe de base abstraite pour tous les wrappers de modèles de segmentation.

Chaque modèle DOIT implémenter :
  - name         : Nom lisible du modèle.
  - load()       : Chargement en mémoire (poids, session ONNX, etc.).
  - predict()    : Inférence sur une frame BGR → masque float [0,1].
  - cleanup()    : Libération des ressources GPU/mémoire.
  - get_flops()  : Estimation ou mesure des FLOPs par frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModelWrapper(ABC):
    """Interface commune pour tous les modèles de Video Matting."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom lisible du modèle (ex: 'MediaPipe Portrait')."""
        ...

    @property
    def input_size(self) -> tuple[int, int] | None:
        """
        Taille d'entrée attendue (H, W). None si la taille est dynamique.
        Utilisé pour le redimensionnement automatique avant inférence.
        """
        return None

    @abstractmethod
    def load(self) -> None:
        """
        Charge le modèle en mémoire.

        Cette méthode est appelée une seule fois avant la boucle d'inférence.
        Elle doit télécharger les poids si nécessaire et initialiser la session.
        """
        ...

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Exécute l'inférence sur une frame BGR.

        L'implémentation doit gérer le pre-processing (resize, normalisation)
        et le post-processing (seuillage, resize vers taille originale).

        Args:
            frame_bgr: Image BGR (H, W, 3) en uint8.

        Returns:
            Masque de segmentation (H, W) en float32 [0, 1].
            H,W doivent correspondre aux dimensions de la frame d'entrée.
        """
        ...

    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """
        Exécute l'inférence sur un lot de frames BGR.

        Par défaut, cette méthode boucle sur predict().
        Elle devrait être surchargée pour les modèles supportant le batching natif
        (ex: ONNX, PyTorch) afin de maximiser l'utilisation du GPU.

        Args:
            frames_bgr: Liste de frames BGR (H, W, 3) en uint8.

        Returns:
            Liste de masques (H, W) en float32 [0, 1].
        """
        return [self.predict(f) for f in frames_bgr]

    @abstractmethod
    def get_flops(self, input_shape: tuple[int, int, int] = (3, 256, 256)) -> float:
        """
        Retourne le nombre de FLOPs pour une inférence.

        Args:
            input_shape: Shape de l'input (C, H, W).

        Returns:
            Nombre de FLOPs (Floating Point Operations). -1 si non mesurable.
        """
        ...

    def reset_state(self) -> None:  # noqa: B027 — no-op par défaut : seuls les modèles récurrents (RVM) doivent l'override.
        """
        Réinitialise l'état interne du modèle (pour les modèles récurrents
        comme RVM qui maintiennent un état entre les frames).

        Appelé au début de chaque vidéo.
        """
        pass

    def cleanup(self) -> None:  # noqa: B027 — no-op par défaut : seuls les modèles avec ressources lourdes (sessions ONNX, GPU) doivent l'override.
        """
        Libère les ressources (GPU, sessions ONNX, etc.).

        Appelé après la fin du benchmark pour ce modèle.
        """
        pass

    def __repr__(self) -> str:
        """Return a human-readable string representation."""
        return f"<{self.__class__.__name__} '{self.name}'>"
