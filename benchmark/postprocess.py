"""
Moteur de post-processing pour les masques de segmentation.

Modèles supportés avec des méthodes adaptées :
  - mediapipe_portrait  : gaussian_blur, morphological_close, morphological_open,
                          bilateral_filter, temporal_ema, feathering
  - mobilenetv3_lraspp  : gaussian_blur, morphological_close, morphological_open,
                          guided_filter, largest_components, hole_filling
  - rvm                 : gaussian_blur, morphological_close, alpha_gamma, temporal_ema
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Répertoire de stockage des configs JSON
POSTPROCESS_CONFIGS_DIR = Path(__file__).parent / "postprocess_configs"

# Clés des modèles supportés (doivent correspondre aux clés de MODEL_REGISTRY)
SUPPORTED_MODELS = ["mediapipe_portrait", "mobilenetv3_lraspp", "rvm"]

# ─── Définition des méthodes disponibles par modèle ───────────────────────────
# Chaque méthode possède : label (UI), params avec valeur par défaut + (min, max, step)
METHODS_BY_MODEL: Dict[str, List[Dict[str, Any]]] = {
    "mediapipe_portrait": [
        {
            "name": "gaussian_blur",
            "label": "Gaussian Blur",
            "description": "Lisse les bords du masque avant seuillage",
            "params": {
                "ksize": {"default": 5, "min": 3, "max": 15, "step": 2, "type": "int"},
                "sigma": {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1, "type": "float"},
            },
        },
        {
            "name": "morphological_close",
            "label": "Fermeture morphologique",
            "description": "Bouche les petits trous dans le foreground",
            "params": {
                "ksize": {"default": 5, "min": 3, "max": 15, "step": 2, "type": "int"},
            },
        },
        {
            "name": "morphological_open",
            "label": "Ouverture morphologique",
            "description": "Supprime les petits artefacts isolés",
            "params": {
                "ksize": {"default": 3, "min": 3, "max": 9, "step": 2, "type": "int"},
            },
        },
        {
            "name": "bilateral_filter",
            "label": "Filtre bilatéral",
            "description": "Lissage en préservant les bords nets",
            "params": {
                "d": {"default": 9, "min": 5, "max": 15, "step": 2, "type": "int"},
                "sigma_color": {"default": 50, "min": 25, "max": 100, "step": 5, "type": "int"},
                "sigma_space": {"default": 50, "min": 25, "max": 100, "step": 5, "type": "int"},
            },
        },
        {
            "name": "temporal_ema",
            "label": "Lissage temporel (EMA)",
            "description": "Réduit le flickering entre frames consécutives",
            "params": {
                "alpha": {"default": 0.7, "min": 0.5, "max": 0.95, "step": 0.05, "type": "float"},
            },
        },
        {
            "name": "feathering",
            "label": "Feathering (bords doux)",
            "description": "Applique un flou gaussien uniquement près des bords",
            "params": {
                "ksize": {"default": 11, "min": 5, "max": 21, "step": 2, "type": "int"},
            },
        },
        {
            "name": "binarize",
            "label": "Seuil de binarisation",
            "description": "Binarise le masque final (pixel ≥ seuil → 1.0, sinon 0.0)",
            "params": {
                "threshold": {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05, "type": "float"},
            },
        },
    ],
    "mobilenetv3_lraspp": [
        {
            "name": "gaussian_blur",
            "label": "Gaussian Blur",
            "description": "Adoucit le masque",
            "params": {
                "ksize": {"default": 5, "min": 3, "max": 15, "step": 2, "type": "int"},
                "sigma": {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1, "type": "float"},
            },
        },
        {
            "name": "morphological_close",
            "label": "Fermeture morphologique",
            "description": "Bouche les trous (vêtements, cheveux)",
            "params": {
                "ksize": {"default": 7, "min": 3, "max": 15, "step": 2, "type": "int"},
            },
        },
        {
            "name": "morphological_open",
            "label": "Ouverture morphologique",
            "description": "Supprime les faux positifs en arrière-plan",
            "params": {
                "ksize": {"default": 3, "min": 3, "max": 9, "step": 2, "type": "int"},
            },
        },
        {
            "name": "guided_filter",
            "label": "Filtre guidé",
            "description": "Affine les bords en utilisant la frame source",
            "params": {
                "radius": {"default": 4, "min": 2, "max": 8, "step": 1, "type": "int"},
                "eps": {"default": 0.05, "min": 0.01, "max": 0.1, "step": 0.01, "type": "float"},
            },
        },
        {
            "name": "largest_components",
            "label": "Garder les N plus grands blobs",
            "description": "Supprime les petits blobs isolés, garde les N plus grands",
            "params": {
                "n": {"default": 1, "min": 1, "max": 3, "step": 1, "type": "int"},
            },
        },
        {
            "name": "hole_filling",
            "label": "Remplissage des trous",
            "description": "Bouche les trous entièrement entourés de foreground",
            "params": {},
        },
        {
            "name": "binarize",
            "label": "Seuil de binarisation",
            "description": "Binarise le masque final (pixel ≥ seuil → 1.0, sinon 0.0)",
            "params": {
                "threshold": {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05, "type": "float"},
            },
        },
    ],
    "rvm": [
        {
            "name": "gaussian_blur",
            "label": "Gaussian Blur",
            "description": "Adoucit l'alpha matte",
            "params": {
                "ksize": {"default": 5, "min": 3, "max": 15, "step": 2, "type": "int"},
                "sigma": {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1, "type": "float"},
            },
        },
        {
            "name": "morphological_close",
            "label": "Fermeture morphologique",
            "description": "Améliore les zones flottantes de l'alpha",
            "params": {
                "ksize": {"default": 5, "min": 3, "max": 15, "step": 2, "type": "int"},
            },
        },
        {
            "name": "alpha_gamma",
            "label": "Correction gamma alpha",
            "description": "Accentue ou atténue le contraste de l'alpha matte",
            "params": {
                "gamma": {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "type": "float"},
            },
        },
        {
            "name": "temporal_ema",
            "label": "Lissage temporel (EMA)",
            "description": "Lissage temporel supplémentaire sur l'alpha",
            "params": {
                "alpha": {"default": 0.7, "min": 0.5, "max": 0.95, "step": 0.05, "type": "float"},
            },
        },
        {
            "name": "binarize",
            "label": "Seuil de binarisation",
            "description": "Binarise le masque final (pixel ≥ seuil → 1.0, sinon 0.0)",
            "params": {
                "threshold": {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05, "type": "float"},
            },
        },
    ],
}


# ─── Fonctions de post-processing individuelles ───────────────────────────────

def _gaussian_blur(mask: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    ksize = int(ksize) | 1  # garantir impair
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma)


def _morphological_close(mask: np.ndarray, ksize: int) -> np.ndarray:
    ksize = int(ksize) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _morphological_open(mask: np.ndarray, ksize: int) -> np.ndarray:
    ksize = int(ksize) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _bilateral_filter(
    mask: np.ndarray,
    frame_bgr: Optional[np.ndarray],
    d: int,
    sigma_color: int,
    sigma_space: int,
) -> np.ndarray:
    mask_u8 = (mask * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(mask_u8, int(d), float(sigma_color), float(sigma_space))
    return filtered.astype(np.float32) / 255.0


def _temporal_ema(mask: np.ndarray, prev_mask: Optional[np.ndarray], alpha: float) -> np.ndarray:
    if prev_mask is None:
        return mask
    return alpha * mask + (1.0 - alpha) * prev_mask


def _guided_filter(
    mask: np.ndarray,
    frame_bgr: Optional[np.ndarray],
    radius: int,
    eps: float,
) -> np.ndarray:
    if frame_bgr is None:
        return mask
    try:
        guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Resize guide to match mask if needed
        if guide.shape != mask.shape:
            guide = cv2.resize(guide, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Use ximgproc if available
        guided = cv2.ximgproc.guidedFilter(guide, mask, int(radius), float(eps))
        return np.clip(guided, 0.0, 1.0)
    except (AttributeError, cv2.error):
        # ximgproc not available — fallback to joint bilateral
        return mask


def _largest_components(mask: np.ndarray, n: int) -> np.ndarray:
    bin_mask = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # Sort components by area (exclude background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    top_n_labels = np.argsort(areas)[::-1][:int(n)] + 1  # +1 to skip background
    result = np.zeros_like(bin_mask)
    for label in top_n_labels:
        result[labels == label] = 1
    return result.astype(np.float32)


def _hole_filling(mask: np.ndarray) -> np.ndarray:
    bin_mask = (mask > 0.5).astype(np.uint8)
    # Flood fill from corners to find background
    h, w = bin_mask.shape
    flood = bin_mask.copy()
    flood_inv = cv2.bitwise_not(flood)
    seed = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_inv, seed, (0, 0), 255)
    flood_inv_flipped = cv2.bitwise_not(flood_inv)
    filled = cv2.bitwise_or(bin_mask, flood_inv_flipped)
    return filled.astype(np.float32)


def _feathering(mask: np.ndarray, ksize: int) -> np.ndarray:
    ksize = int(ksize) | 1
    bin_mask = (mask > 0.5).astype(np.uint8)
    # Extract edge zone via dilation - erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize // 2 + 1, ksize // 2 + 1))
    dilated = cv2.dilate(bin_mask, kernel)
    eroded = cv2.erode(bin_mask, kernel)
    edge_zone = (dilated - eroded).astype(bool)
    # Apply blur only in edge zone
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    result = mask.copy()
    result[edge_zone] = blurred[edge_zone]
    return result


def _alpha_gamma(mask: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(mask, 0.0, 1.0), float(gamma)).astype(np.float32)


def _binarize(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask >= float(threshold)).astype(np.float32)


# ─── Classe PostProcessor ─────────────────────────────────────────────────────

class PostProcessor:
    """
    Applique une chaîne de post-processing sur les masques float32 [0, 1].

    Usage :
        pp = PostProcessor(config)
        pp.reset()  # au début de chaque vidéo
        mask = pp.apply(mask, frame_bgr=frame)
    """

    def __init__(self, config: Dict):
        self._config = config
        self._prev_mask: Optional[np.ndarray] = None
        self._methods: List[Dict] = config.get("methods", [])

    def reset(self) -> None:
        """Réinitialise l'état temporel (à appeler au début de chaque vidéo)."""
        self._prev_mask = None

    def apply(self, mask: np.ndarray, frame_bgr: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applique les méthodes activées dans l'ordre de la config.

        Args:
            mask      : float32 mask [0, 1] de forme (H, W) ou (H, W, 1)
            frame_bgr : frame source en BGR (optionnel, utilisé par bilateral/guided)

        Returns:
            float32 mask [0, 1] de forme (H, W)
        """
        m = mask.squeeze().astype(np.float32)

        for method_cfg in self._methods:
            if not method_cfg.get("enabled", False):
                continue
            name = method_cfg.get("name", "")
            params = method_cfg.get("params", {})

            try:
                if name == "gaussian_blur":
                    m = _gaussian_blur(m, params.get("ksize", 5), params.get("sigma", 1.0))

                elif name == "morphological_close":
                    m = _morphological_close(m, params.get("ksize", 5))

                elif name == "morphological_open":
                    m = _morphological_open(m, params.get("ksize", 3))

                elif name == "bilateral_filter":
                    m = _bilateral_filter(
                        m, frame_bgr,
                        params.get("d", 9),
                        params.get("sigma_color", 50),
                        params.get("sigma_space", 50),
                    )

                elif name == "temporal_ema":
                    m = _temporal_ema(m, self._prev_mask, params.get("alpha", 0.7))

                elif name == "guided_filter":
                    m = _guided_filter(m, frame_bgr, params.get("radius", 4), params.get("eps", 0.05))

                elif name == "largest_components":
                    m = _largest_components(m, params.get("n", 1))

                elif name == "hole_filling":
                    m = _hole_filling(m)

                elif name == "feathering":
                    m = _feathering(m, params.get("ksize", 11))

                elif name == "alpha_gamma":
                    m = _alpha_gamma(m, params.get("gamma", 1.0))

                elif name == "binarize":
                    m = _binarize(m, params.get("threshold", 0.5))

                else:
                    logger.warning("Méthode de post-process inconnue : %s", name)

            except Exception as e:
                logger.warning("Erreur dans post-process '%s' : %s", name, e)

        # Mettre à jour l'état EMA
        self._prev_mask = m.copy()
        return np.clip(m, 0.0, 1.0)

    @property
    def has_active_methods(self) -> bool:
        return any(m.get("enabled", False) for m in self._methods)


# ─── Persistence des configs ──────────────────────────────────────────────────

def load_config(model_key: str) -> Dict:
    """Charge la config JSON pour un modèle. Retourne un dict vide si inexistant."""
    path = POSTPROCESS_CONFIGS_DIR / f"{model_key}.json"
    if not path.exists():
        return _default_config(model_key)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Impossible de lire la config post-process '%s' : %s", path, e)
        return _default_config(model_key)


def save_config(model_key: str, config: Dict) -> None:
    """Sauvegarde la config JSON pour un modèle."""
    POSTPROCESS_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = POSTPROCESS_CONFIGS_DIR / f"{model_key}.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Config post-process sauvegardée : %s", path)


def _default_config(model_key: str) -> Dict:
    """Construit une config par défaut (toutes les méthodes désactivées)."""
    methods = METHODS_BY_MODEL.get(model_key, [])
    return {
        "model_key": model_key,
        "methods": [
            {
                "name": m["name"],
                "enabled": False,
                "params": {k: v["default"] for k, v in m["params"].items()},
            }
            for m in methods
        ],
    }


def get_postprocessor(model_key: str) -> Optional[PostProcessor]:
    """
    Retourne un PostProcessor pour ce modèle s'il est supporté et configuré,
    None sinon (ou si aucune méthode n'est activée).
    """
    if model_key not in SUPPORTED_MODELS:
        return None
    config = load_config(model_key)
    pp = PostProcessor(config)
    return pp if pp.has_active_methods else None
