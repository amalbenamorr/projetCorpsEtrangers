"""
tool_anomaly_hunter.py
──────────────────────
Répond à : EST-CE NORMAL ?
Compare avec les 4730 images de viande propre.
Score 0 = identique à la viande propre. Score élevé = anomalie.

FIX: Seuils recalibrés. Score 0.5 sur viande propre = bug de calibration.
La distance cosinus moyenne dans la memory bank propre est ~0.05-0.15.
Seuil conservateur recalibré à 0.25 (pas 0.08).
"""

import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path

FEATURES_PATH = "models/patchcore/features.npy"
BACKBONE_PATH = "models/patchcore/backbone.pt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at import ─────────────────────────────────────────────────
_backbone          = models.resnet18(weights=None)
_feature_extractor = nn.Sequential(*list(_backbone.children())[:-2])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _feature_extractor.load_state_dict(
        torch.load(BACKBONE_PATH, map_location=DEVICE, weights_only=True)
    )

_feature_extractor = _feature_extractor.to(DEVICE).eval()
_memory_bank       = np.load(FEATURES_PATH)

# Calibrate thresholds from memory bank distribution
# Compute pairwise distances within clean meat → find natural spread
_sample_size = min(500, len(_memory_bank))
_sample_idx  = np.random.choice(len(_memory_bank), _sample_size, replace=False)
_sample      = _memory_bank[_sample_idx]
_probe       = _memory_bank[np.random.choice(len(_memory_bank), 50, replace=False)]
_calib_dists = cosine_distances(_probe, _sample)
_calib_min   = float(np.mean(np.min(_calib_dists, axis=1)))
_calib_std   = float(np.std(np.min(_calib_dists, axis=1)))

# Dynamic thresholds: anomaly = mean + 2*std of clean distribution
THRESHOLD_ANOMALY = float(np.clip(_calib_min + 3.0 * _calib_std, 0.15, 0.60))
THRESHOLD_CERTAIN = float(np.clip(_calib_min + 5.0 * _calib_std, 0.25, 0.80))

print(f"[AnomalyHunter] Calibrated: clean_mean={_calib_min:.4f} std={_calib_std:.4f} "
      f"| threshold_anomaly={THRESHOLD_ANOMALY:.4f} threshold_certain={THRESHOLD_CERTAIN:.4f}")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def tool_anomaly_hunter(image_input) -> dict:
    """
    Répond à : EST-CE NORMAL ?
    
    Returns:
        score              : float 0→1 (distance cosinus au plus proche voisin propre)
        is_anomaly         : bool (seuil dynamique calibré sur la distribution propre)
        confidence         : "high" / "medium" / "low"
        relative_distance  : float (score / moyenne de la distribution propre)
        threshold_used     : float (seuil dynamique)
        verdict            : str
    """
    # ── Load image ────────────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        # Handle single-channel patches
        if len(image_input.shape) == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        img = Image.fromarray(image_input[:, :, ::-1])  # BGR → RGB
    else:
        img = image_input

    # ── Feature extraction ────────────────────────────────────────────────
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = _feature_extractor(tensor)
        feat = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()

    # ── Distance to memory bank ───────────────────────────────────────────
    distances  = cosine_distances([feat], _memory_bank)[0]
    score      = float(np.min(distances))
    mean_dist  = float(np.mean(distances))

    # Relative distance (how far compared to clean meat spread)
    relative = score / (_calib_min + 1e-8)

    # ── Confidence based on calibrated thresholds ─────────────────────────
    if score < THRESHOLD_ANOMALY:
        confidence = "low"       # inside clean distribution
        is_anomaly = False
    elif score < THRESHOLD_CERTAIN:
        confidence = "medium"    # borderline
        is_anomaly = True
    else:
        confidence = "high"      # clearly anomalous
        is_anomaly = True

    return {
        "score":            round(score, 4),
        "is_anomaly":       is_anomaly,
        "confidence":       confidence,
        "relative_distance": round(relative, 4),
        "threshold_used":   round(THRESHOLD_ANOMALY, 4),
        "clean_mean":       round(_calib_min, 4),
        "verdict":          f"anomaly_score={score:.4f} is_anomaly={is_anomaly} confidence={confidence}"
    }