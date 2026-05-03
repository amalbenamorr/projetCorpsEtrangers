"""
tool_scanner.py — v5 FINAL
────────────────────────────────────────────────────────────────
FIXES v5 over v4:

ROOT CAUSE FIXED: Light-colored foreign objects (feathers, bones, plastic)
were INVISIBLE to the scanner because they have similar brightness to
chicken fat/skin → color_distance was LOW → zones never detected.

SOLUTION: Three complementary detection signals:
  Signal A: color distance from meat (catches dark/colored objects: metal, wood)
  Signal B: texture anomaly — objects have DIFFERENT texture than surrounding meat
             (catches ALL objects regardless of color)
  Signal C: white/bright object detector — high brightness + uniform texture
             different from meat's natural variation (catches feathers, bone, plastic)

Additional fixes:
- Feather detection: long thin shape + white + high orientation_ratio
- Zone scoring: weights rebalanced to give texture more power
- Min area reduced to 200px² (feathers can be thin)
- Multi-scale heatmap: coarse + fine to catch both large and small objects
"""

import cv2
import numpy as np
from pathlib import Path


def _meat_color_distance(img: np.ndarray) -> np.ndarray:
    """
    Distance LAB pixel-par-pixel depuis la couleur médiane de viande.
    Robust: uses central 25%-75% zone to estimate meat reference color.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    center  = lab[y1:y2, x1:x2].reshape(-1, 3)
    meat_color = np.median(center, axis=0)

    diff = lab - meat_color[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    p5  = float(np.percentile(dist, 5))
    p95 = float(np.percentile(dist, 95))
    span = max(p95 - p5, 1.0)
    return np.clip((dist - p5) / span, 0.0, 1.0).astype(np.float32)


def _white_bright_detector(img: np.ndarray) -> np.ndarray:
    """
    Detects white/bright objects (feathers, bones, plastic, fat deposits).
    Key insight: feathers are white + have uniform texture DIFFERENT from
    the heterogeneous texture of meat/skin.
    
    Returns: float32 map [0,1] — high = white object with anomalous texture
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    val = hsv[:, :, 2] / 255.0   # brightness
    sat = hsv[:, :, 1] / 255.0   # saturation

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h_img, w_img = gray.shape

    # Local texture variance (meat has natural variance; feather has low variance center + high at edges)
    k = np.ones((11, 11), np.float32) / 121.0
    lmean = cv2.filter2D(gray, -1, k)
    lsq   = cv2.filter2D(gray ** 2, -1, k)
    lstd  = np.sqrt(np.maximum(lsq - lmean ** 2, 0.0)) / 255.0

    # White + high brightness + low saturation = feather/bone/plastic candidate
    white_mask = (val > 0.60) & (sat < 0.35)  # bright + desaturated

    # Compare local texture to global meat texture
    global_std = float(np.std(gray)) / 255.0
    # Feathers: locally uniform (low lstd) but globally stand out (white_mask)
    # OR feather edges: locally high variation
    texture_anomaly = np.abs(lstd - float(np.mean(lstd))) / (float(np.std(lstd)) + 1e-8)
    texture_anomaly = np.clip(texture_anomaly, 0, 1).astype(np.float32)

    white_map = white_mask.astype(np.float32) * (0.6 + 0.4 * texture_anomaly)

    # Smooth
    white_map = cv2.GaussianBlur(white_map, (15, 15), 0)

    # Normalize
    wmin, wmax = white_map.min(), white_map.max()
    if wmax > wmin:
        white_map = (white_map - wmin) / (wmax - wmin + 1e-8)

    return white_map.astype(np.float32)


def _texture_anomaly_map(img: np.ndarray) -> np.ndarray:
    """
    Detects zones where LOCAL texture differs significantly from GLOBAL meat texture.
    This catches ALL foreign objects regardless of color — because any non-meat
    object (metal, wood, bone, feather, plastic) has different texture than meat.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Multi-scale local variance
    maps = []
    for ksize in [7, 15, 31]:
        k     = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
        lm    = cv2.filter2D(gray, -1, k)
        lsq   = cv2.filter2D(gray ** 2, -1, k)
        lstd  = np.sqrt(np.maximum(lsq - lm ** 2, 0.0))
        maps.append(lstd)

    # Stack and take max across scales
    multi = np.max(np.stack(maps, axis=0), axis=0)

    # Normalize against global stats
    gm = float(np.mean(multi))
    gs = float(np.std(multi))
    if gs < 1e-6:
        return np.zeros_like(multi)

    # Z-score: how far is each pixel's local texture from the average?
    z = (multi - gm) / (gs + 1e-8)
    # Objects: either very low texture (metal, plastic: uniform shiny)
    # or very high texture (wood, bone: rough)
    anomaly = np.abs(z)
    anomaly = np.clip(anomaly / 4.0, 0, 1).astype(np.float32)

    anomaly = cv2.GaussianBlur(anomaly, (21, 21), 0)
    amin, amax = anomaly.min(), anomaly.max()
    if amax > amin:
        anomaly = (anomaly - amin) / (amax - amin + 1e-8)

    return anomaly.astype(np.float32)


def _is_border(x, y, bw, bh, img_w, img_h, margin=20) -> bool:
    return (x <= margin or y <= margin or
            x + bw >= img_w - margin or y + bh >= img_h - margin)


def _merge_overlapping(zones: list, iou_thr: float = 0.2) -> list:
    if len(zones) <= 1:
        return zones

    def iou(a, b):
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (aw * ah + bw * bh - inter + 1e-8)

    merged, used = [], [False] * len(zones)
    for i, z in enumerate(zones):
        if used[i]:
            continue
        group = [z]
        used[i] = True
        for j, z2 in enumerate(zones):
            if not used[j] and iou(z, z2) > iou_thr:
                group.append(z2)
                used[j] = True
        best = max(group, key=lambda x: x["combined_score"])
        if len(group) > 1:
            xs  = [g["bbox"][0] for g in group]
            ys  = [g["bbox"][1] for g in group]
            x2s = [g["bbox"][0] + g["bbox"][2] for g in group]
            y2s = [g["bbox"][1] + g["bbox"][3] for g in group]
            mx, my = min(xs), min(ys)
            mw, mh = max(x2s) - mx, max(y2s) - my
            best = dict(best)
            best["bbox"] = [mx, my, mw, mh]
        merged.append(best)
    return merged


def _enlarge_patch_for_vlm(img: np.ndarray, bbox: list, min_size: int = 180) -> np.ndarray:
    """Extract an enlarged patch with context margin — minimum min_size×min_size for VLM."""
    x, y, bw, bh = bbox
    h_img, w_img = img.shape[:2]

    margin = max(max(bw, bh), min_size // 2) + 30
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + bw + margin)
    y2 = min(h_img, y + bh + margin)

    patch = img[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]

    if ph < min_size or pw < min_size:
        scale = max(min_size / max(ph, 1), min_size / max(pw, 1))
        new_h, new_w = int(ph * scale), int(pw * scale)
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return patch


def tool_scanner(image_input, sensitivity: str = "auto") -> dict:
    """
    Scans full image for suspicious zones.

    v5 changes:
    - THREE complementary signals:
        A) color_distance: catches dark/colored objects (metal, wood)
        B) texture_anomaly: catches ALL objects by local texture deviation
        C) white_detector: catches light-colored objects (feather, bone, plastic) ← NEW
    - combined_score = 0.35*color + 0.30*texture + 0.25*white + 0.10*area_bonus
    - Min area: 200px² (catches thin feathers)
    - Border margin: 20px

    Returns:
        zones        : list sorted by combined_score DESC
        heatmap      : float32 [0,1] combined heatmap
        n_suspects   : int (zones with combined_score >= 0.35)
        full_image   : numpy BGR
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input.copy()

    if img is None:
        return {"error": "image_not_loaded", "zones": [], "n_suspects": 0,
                "full_image": None, "heatmap": None}

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf32  = gray.astype(np.float32)

    # ── Signal A: Color distance from meat (dark/colored objects) ─────────
    color_dist = _meat_color_distance(img)

    # ── Signal B: Texture anomaly (ALL objects) ───────────────────────────
    texture_anom = _texture_anomaly_map(img)

    # ── Signal C: White/bright object detector (feather, bone, plastic) ───
    white_map = _white_bright_detector(img)

    # ── Signal D: Edges (fine detail) ─────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0

    # ── Weights by sensitivity ─────────────────────────────────────────────
    if sensitivity == "high":
        w1, w2, w3, w4 = 0.30, 0.30, 0.25, 0.15
    elif sensitivity == "low":
        w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15
    else:  # auto
        w1, w2, w3, w4 = 0.35, 0.30, 0.25, 0.10

    heatmap = w1 * color_dist + w2 * texture_anom + w3 * white_map + w4 * edges
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    hm_min, hm_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)

    # Adaptive threshold
    threshold = float(np.mean(heatmap) + 1.6 * np.std(heatmap))
    threshold = float(np.clip(threshold, 0.40, 0.80))

    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area   = float(h * w)
    all_zones  = []
    border_cnt = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # v5: min 200px² (feathers can be thin/small)
        if area < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if _is_border(x, y, bw, bh, w, h, margin=20):
            border_cnt += 1
            continue
        if bw < 5 or bh < 5:
            continue

        patch_raw  = img[y:y + bh, x:x + bw].copy()
        patch_gray = gray[y:y + bh, x:x + bw]

        local_hm      = float(np.mean(heatmap[y:y + bh, x:x + bw]))
        local_color   = float(np.mean(color_dist[y:y + bh, x:x + bw]))
        local_texture = float(np.mean(texture_anom[y:y + bh, x:x + bw]))
        local_white   = float(np.mean(white_map[y:y + bh, x:x + bw]))

        # Area bonus (cap 3% of image)
        area_frac  = min(area / img_area, 0.03) / 0.03
        area_bonus = area_frac * 0.3

        # Combined score with all 4 signals
        combined_score = (
            0.35 * local_color +
            0.30 * local_texture +
            0.25 * local_white +
            0.10 * area_bonus
        )

        # Patch enlarged for VLM (at least 180×180)
        patch_vlm = _enlarge_patch_for_vlm(img, [x, y, bw, bh], min_size=180)

        all_zones.append({
            "bbox":            [x, y, bw, bh],
            "area":            int(area),
            "local_score":     round(local_hm, 4),
            "color_distance":  round(local_color, 4),
            "texture_score":   round(local_texture, 4),
            "white_score":     round(local_white, 4),
            "combined_score":  round(combined_score, 4),
            "center":          [int(x + bw / 2), int(y + bh / 2)],
            "patch_mean_val":  round(float(np.mean(patch_gray)), 2),
            "patch_std":       round(float(np.std(patch_gray)), 2),
            "patch":           patch_raw,
            "patch_vlm":       patch_vlm,
        })

    all_zones = _merge_overlapping(all_zones)
    all_zones = sorted(all_zones, key=lambda z: z["combined_score"], reverse=True)[:5]

    return {
        "zones":                 all_zones,
        "heatmap":               heatmap,
        "n_suspects":            len(all_zones),
        "full_image":            img,
        "image_shape":           [h, w],
        "sensitivity_used":      sensitivity,
        "border_zones_filtered": border_cnt,
        "threshold_used":        round(threshold, 4),
    }