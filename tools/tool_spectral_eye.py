"""
tool_spectral_eye.py — v4 FINAL
────────────────────────────────
Répond à : C'EST QUOI COMME MATIÈRE ?

FIXES v4 over v3:
- Wood signature: brun + texture fibreuse + FFT directionnelle MODÉRÉE
  (bois a des fibres mais moins directionnels que plume)
- Feather: blanc/gris TRÈS clair + fibres TRÈS directionnelles + FFT haute fréquence
- Bone: blanc cassé + DENSE + peu de direction + brillance faible
- Metal: gris/argent + réflectance ÉLEVÉE + non-directionnel + bords nets
- Plastic: couleur uniforme + FFT moyenne + NO directionnel
- Better separation: wood vs feather (white_ratio is key discriminator)
- Added "wood" explicitly as a material score
- density_score recalibrated (Sobel std not raw density)
"""

import cv2
import numpy as np
from pathlib import Path


def _color_profile(img_bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L   = lab[:, :, 0] / 255.0
    a   = (lab[:, :, 1] - 128) / 128.0
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    b_ch = img_bgr[:, :, 0].astype(np.float32)
    g_ch = img_bgr[:, :, 1].astype(np.float32)
    r_ch = img_bgr[:, :, 2].astype(np.float32)

    # Brown ratio: brownish = moderate red + moderate green, low blue
    brown_ratio = float(np.mean(
        (r_ch > 80) & (g_ch > 50) & (b_ch < 120) &
        (r_ch > g_ch) & (g_ch > b_ch)
    ))

    return {
        "mean_L":       float(np.mean(L)),
        "mean_sat":     float(np.mean(sat)),
        "mean_val":     float(np.mean(val)),
        "std_sat":      float(np.std(sat)),
        "mean_a":       float(np.mean(a)),
        "white_ratio":  float(np.mean((val > 0.72) & (sat < 0.22))),
        "red_ratio":    float(np.mean(r_ch)) / (float(np.mean(b_ch)) + 1e-8),
        "brown_ratio":  brown_ratio,
        "gray_ratio":   float(np.mean((sat < 0.15) & (val > 0.2) & (val < 0.8))),
        "dark_ratio":   float(np.mean(val < 0.25)),
    }


def tool_spectral_eye(image_input) -> dict:
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input.copy()

    if img is None:
        return {"error": "cannot load image", "dominant_material": "unknown"}

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h_orig, w_orig = img.shape[:2]
    if h_orig < 16 or w_orig < 16:
        img = cv2.resize(img, (max(w_orig, 64), max(h_orig, 64)), interpolation=cv2.INTER_CUBIC)

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_u8  = gray
    gray_f32 = gray.astype(np.float32)
    gray_f64 = gray.astype(np.float64)
    h, w     = gray.shape

    # ── Color profile ──────────────────────────────────────────────────────
    color = _color_profile(img)

    # ── FFT analysis ────────────────────────────────────────────────────────
    fshift    = np.fft.fftshift(np.fft.fft2(gray_f64))
    magnitude = np.log1p(np.abs(fshift))
    cy, cx    = h // 2, w // 2
    r_low     = max(2, min(h, w) // 8)
    r_high    = max(4, min(h, w) // 4)
    Y, X      = np.ogrid[:h, :w]
    d_map     = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    total_e   = float(np.sum(magnitude)) + 1e-8
    energy_low   = float(np.sum(magnitude[d_map < r_low]))  / total_e
    energy_high  = float(np.sum(magnitude[d_map > r_high])) / total_e
    energy_mid   = 1.0 - energy_low - energy_high

    # Directional asymmetry
    mh = magnitude.copy(); mh[d_map <= r_high] = 0
    q  = [float(np.sum(mh[s1, s2])) for s1, s2 in
          [(slice(None, cy), slice(None, cx)), (slice(None, cy), slice(cx, None)),
           (slice(cy, None), slice(None, cx)), (slice(cy, None), slice(cx, None))]]
    q  = np.array(q) + 1e-8
    q /= q.sum()
    directional = float(q.max() - q.min())

    # ── Texture ────────────────────────────────────────────────────────────
    k          = np.ones((15, 15), np.float32) / 225.0
    lmean      = cv2.filter2D(gray_f32, -1, k)
    lsq        = cv2.filter2D(gray_f32 ** 2, -1, k)
    lstd       = np.sqrt(np.maximum(lsq - lmean ** 2, 0.0))
    uniformity = float(np.clip(1.0 - np.std(lstd) / (np.mean(lstd) + 1e-8), 0, 1))

    # Fiber orientation ratio
    kh  = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], np.float32)
    kv  = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], np.float32)
    kd1 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], np.float32)
    kd2 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], np.float32)
    resps = [np.abs(cv2.filter2D(gray_f32, -1, k2)) for k2 in [kh, kv, kd1, kd2]]
    orient_ratio = float(np.mean(np.max(resps, axis=0) / (np.mean(resps, axis=0) + 1e-8)))

    # ── Laplacian + Sobel ──────────────────────────────────────────────────
    lap_mean  = float(np.mean(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray_u8, (9,9), 0), cv2.CV_64F))))
    sx        = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    sy        = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    density   = np.sqrt(sx ** 2 + sy ** 2)
    density_s = float(np.std(density) / (np.mean(density) + 1e-8))

    blur_s    = cv2.GaussianBlur(gray_f32, (31, 31), 0)
    reflect_v = float(np.std(gray_f32 - blur_s) / (np.mean(gray_f32) + 1e-8))

    lap_n = float(np.clip(lap_mean / 50.0, 0, 1))
    den_n = float(np.clip(density_s / 3.0, 0, 1))

    # ────────────────────────────────────────────────────────────────────────
    # RECALIBRATED MATERIAL SIGNATURES v4
    # KEY DISCRIMINATORS:
    # white_ratio: feather=HIGH, bone=MODERATE, wood=LOW, metal=VERY LOW
    # brown_ratio: wood=HIGH, others=LOW
    # gray_ratio:  metal=HIGH, bone=MODERATE, others=LOW
    # directional: feather=VERY HIGH, wood=MODERATE, metal=LOW
    # reflect_v:   metal=HIGH, others=LOW
    # ────────────────────────────────────────────────────────────────────────

    # FEATHER: blanc/gris très clair + fibres très directionnelles + hautes fréquences
    feather = max(0.0,
        color["white_ratio"] * 4.0 +          # feather is very white
        directional * 3.0 +                    # strong directional texture
        (orient_ratio - 1.0) * 1.5 +           # fiber orientation
        (1.0 - color["mean_sat"]) * 1.0 +      # low saturation
        energy_high * 1.0 +
        (1.0 - color["brown_ratio"]) * 1.0     # NOT brown
    )

    # WOOD: brun + fibres modérément directionnelles + texture rugueuse
    wood = max(0.0,
        color["brown_ratio"] * 5.0 +           # wood is brown — STRONG signal
        directional * 1.5 +                    # moderately directional (grain)
        (orient_ratio - 1.0) * 1.0 +           # some fiber orientation
        energy_mid * 1.0 +                     # mid frequencies (grain texture)
        (1.0 - color["white_ratio"]) * 1.5 +   # NOT white
        (1.0 - color["gray_ratio"]) * 1.0 +    # NOT gray (not metal)
        den_n * 0.5                            # some texture density
    )

    # METAL: gris/argent + réflectance variable + non-directionnel + bords nets
    metal = max(0.0,
        color["gray_ratio"] * 3.0 +            # metal is gray
        (1.0 - color["mean_sat"]) * 1.5 +      # low saturation
        reflect_v * 2.5 +                      # high reflectance variation
        energy_high * 1.5 +                    # high frequency edges
        (1.0 - directional) * 1.0 +            # isotropic (not directional)
        lap_n * 1.5 +                          # sharp edges
        (1.0 - color["brown_ratio"]) * 1.5 +   # NOT brown
        (1.0 - color["white_ratio"]) * 0.8     # NOT white
    )

    # BONE: blanc cassé + dense + rigide + peu directionnel
    bone = max(0.0,
        (color["mean_L"] - 0.4) * 1.5 +        # bright/light
        color["white_ratio"] * 1.5 +            # moderately white (less than feather)
        (1.0 - color["mean_sat"]) * 1.2 +       # low saturation
        energy_mid * 1.5 +                      # mid frequencies
        den_n * 1.5 +                           # dense/solid
        (1.0 - directional) * 0.8 +             # NOT directional
        (1.0 - color["brown_ratio"]) * 1.0      # NOT brown
    )

    # PLASTIC: couleur uniforme + fréquences moyennes + non-directionnel
    plastic = max(0.0,
        uniformity * 2.0 +                     # very uniform color
        energy_mid * 1.5 +                     # mid frequencies
        (1.0 - directional) * 1.0 +            # NOT directional
        (1.0 - color["std_sat"]) * 1.2 +       # uniform saturation
        energy_high * 0.5
    )

    # ORGANIC/MEAT: rose/rouge + saturé + fréquences basses + doux
    organic = max(0.0,
        color["mean_sat"] * 2.5 +
        color["mean_a"] * 2.5 +                # positive a* = reddish
        energy_low * 2.0 +
        uniformity * 0.5 +
        (1.0 - lap_n) * 0.5 +
        (color["red_ratio"] - 1.0) * 1.0
    )

    raw    = {
        "metal": metal, "bone": bone, "plastic": plastic,
        "feather": feather, "organic": organic, "wood": wood
    }
    tot    = sum(raw.values()) + 1e-8
    scores = {k: round(v / tot, 4) for k, v in raw.items()}
    dominant = max(scores, key=scores.get)

    # ── Subsurface ────────────────────────────────────────────────────────
    sub_prob = float(np.clip(
        (lap_mean / 30.0) * 0.4 +
        float(np.clip(density_s - 0.5, 0, 1)) * 0.4 +
        energy_high * 0.2,
        0, 1
    ))
    evidence = []
    if lap_mean > 15:                   evidence.append("abnormal_laplacian_response")
    if density_s > 1.5:                 evidence.append("abnormal_density_pattern")
    if color["white_ratio"] > 0.25:     evidence.append("light_colored_object")
    if color["brown_ratio"] > 0.20:     evidence.append("brown_object_detected")
    if color["gray_ratio"] > 0.30:      evidence.append("gray_object_detected")

    return {
        "material_signature":  scores,
        "dominant_material":   dominant,
        "color_profile":       {k: round(v, 4) for k, v in color.items()},
        "fft_energy_high":     round(energy_high, 4),
        "fft_energy_low":      round(energy_low, 4),
        "fft_energy_mid":      round(energy_mid, 4),
        "texture_uniformity":  round(uniformity, 4),
        "laplacian_mean":      round(lap_mean, 4),
        "density_score":       round(density_s, 4),
        "directional":         round(directional, 4),
        "orientation_ratio":   round(orient_ratio, 4),
        "subsurface_hint": {
            "subsurface_probability": round(sub_prob, 3),
            "evidence": evidence,
            "confidence": "high" if sub_prob > 0.8 else "medium" if sub_prob > 0.6 else "low"
        },
        "verdict": (
            f"dominant={dominant} "
            f"feather={scores['feather']:.3f} "
            f"metal={scores['metal']:.3f} "
            f"wood={scores['wood']:.3f} "
            f"organic={scores['organic']:.3f}"
        )
    }