"""
tool_object_localizer.py — v4 FINAL
─────────────────────────────────────
FIXES v4 over v3:

ROOT CAUSE OF IMPRECISE BOXES:
- VLM grid donnait des % approximatifs puis snap contour ratait la couleur
- Wood brun sur fond rose → masque couleur trop strict → snap retourne None → fallback VLM estimate → mauvaise bbox

SOLUTION v4:
1. VLM donne DIRECTEMENT x1%,y1%,x2%,y2% (plus précis que center+size)
2. Snap contour avec zone élargie 3× ET fallback multi-stratégie :
   a) masque couleur objet-spécifique
   b) masque "différent du fond" (LAB distance)  
   c) masque Canny edges dans la zone
3. Prend le PLUS GRAND contour valide dans la zone (pas le plus proche du centre)
4. Recalage sur l'image complète en tenant compte du crop_offset du DirectVLM
"""

import base64, re, json, cv2, numpy as np, requests, time, io, os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_VLM_MODEL = os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MAX_DIM = 1024


def _encode(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _call_vlm_bbox(b64: str, media_type: str, obj_type: str, obj_color: str,
                   obj_shape: str, n_objects: int) -> list:
    """
    Demande au VLM de donner des bbox en pourcentage (x1,y1,x2,y2).
    Plus précis que center+size car évite l'ambiguïté sur la taille.
    """
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""You are a precise object localizer. Give bounding box coordinates as percentages.
X: 0=left, 100=right. Y: 0=top, 100=bottom.

Find ALL instances of:
- Type: {obj_type}
- Color: {obj_color}  
- Shape: {obj_shape}
- Expected count: {n_objects}

For EACH instance, give the TIGHT bounding box coordinates (x1=left edge, y1=top edge, x2=right edge, y2=bottom edge).
Be precise — the box should tightly surround the object, not approximate.

Respond ONLY with valid JSON:
{{
  "objects": [
    {{
      "x1_pct": <left edge 0-100>,
      "y1_pct": <top edge 0-100>,
      "x2_pct": <right edge 0-100>,
      "y2_pct": <bottom edge 0-100>,
      "confidence": "<high|medium|low>",
      "description": "<what you see at this location>"
    }}
  ],
  "total_found": <number>
}}

If nothing matches: objects=[] total_found=0"""

    payload = {
        "model": GROQ_VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 600, "temperature": 0.02
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{GROQ_API_BASE}/chat/completions",
                              headers=headers, json=payload, timeout=45)
            if r.status_code == 429:
                time.sleep((2**attempt)*4); continue
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            clean = re.sub(r"```json|```", "", raw).strip()
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                return json.loads(m.group()).get("objects", [])
        except Exception:
            if attempt == 3: break
            time.sleep(2)
    return []


def _pct_to_pixels(x1p, y1p, x2p, y2p, img_w, img_h) -> tuple:
    """Convert percentage coords to pixel coords."""
    x1 = max(0, int(x1p / 100 * img_w))
    y1 = max(0, int(y1p / 100 * img_h))
    x2 = min(img_w, int(x2p / 100 * img_w))
    y2 = min(img_h, int(y2p / 100 * img_h))
    return x1, y1, x2, y2


def _get_color_mask(region: np.ndarray, obj_type: str, obj_color: str) -> np.ndarray:
    """Multi-strategy color mask — falls back to LAB distance if specific mask fails."""
    obj_lower   = (obj_type  or "").lower()
    color_lower = (obj_color or "").lower()

    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sat  = hsv[:, :, 1]
    val  = hsv[:, :, 2]
    b, g, r = region[:, :, 0].astype(np.float32), region[:, :, 1].astype(np.float32), region[:, :, 2].astype(np.float32)

    if "metal" in obj_lower or "gray" in color_lower or "silver" in color_lower or "dark" in color_lower:
        mask = ((sat < 55) & (val > 20) & (val < 185)).astype(np.uint8) * 255
    elif "feather" in obj_lower or "white" in color_lower:
        mask = ((val > 185) & (sat < 50)).astype(np.uint8) * 255
    elif "wood" in obj_lower or "brown" in color_lower:
        # Brun: R > G > B, différence R-B notable
        mask = ((r > 80) & (g > 50) & (b < 130) & (r > g) & (g >= b) & ((r - b) > 15)).astype(np.uint8) * 255
    elif "bone" in obj_lower or "ivory" in color_lower:
        mask = ((val > 165) & (sat < 70)).astype(np.uint8) * 255
    elif "plastic" in obj_lower:
        mask = cv2.Canny(gray, 40, 120)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    elif "rubber" in obj_lower or "black" in color_lower:
        mask = ((val < 65) & (sat < 85)).astype(np.uint8) * 255
    else:
        # Generic: pixels les plus différents du fond
        lab  = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        med  = np.median(lab.reshape(-1, 3), axis=0)
        diff = lab - med[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        thr  = float(np.mean(dist) + 1.0 * np.std(dist))
        mask = (dist > thr).astype(np.uint8) * 255

    return mask


def _snap_to_best_contour(img: np.ndarray, sx1: int, sy1: int, sx2: int, sy2: int,
                           obj_type: str, obj_color: str, min_area: int = 50) -> list:
    """
    Dans la search region, trouve le bbox le plus précis autour de l'objet.
    Stratégie: essaie couleur-spécifique → LAB distance → Canny edges.
    Retourne [x, y, w, h] en coords image complète, ou None.
    """
    region = img[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return None

    rh, rw = region.shape[:2]
    region_area = float(rh * rw)

    masks = []
    # Stratégie 1: masque couleur spécifique
    masks.append(_get_color_mask(region, obj_type, obj_color))

    # Stratégie 2: LAB distance depuis médiane (universel)
    lab  = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    med  = np.median(lab.reshape(-1, 3), axis=0)
    diff = lab - med[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    thr  = float(np.mean(dist) + 0.8 * np.std(dist))
    masks.append((dist > thr).astype(np.uint8) * 255)

    # Stratégie 3: Canny edges
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    masks.append(cv2.dilate(edges, kernel, iterations=2))

    best_bbox = None
    best_area = 0

    for mask in masks:
        k3 = np.ones((3, 3), np.uint8)
        k7 = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        valid = [(cnt, cv2.contourArea(cnt)) for cnt in contours
                 if cv2.contourArea(cnt) >= min_area and cv2.contourArea(cnt) < region_area * 0.80]
        if not valid:
            continue

        # Plus grand contour valide
        best_cnt = max(valid, key=lambda x: x[1])
        area = best_cnt[1]
        if area > best_area:
            best_area = area
            rx, ry, rw_b, rh_b = cv2.boundingRect(best_cnt[0])
            # Padding léger
            pad = 4
            rx = max(0, rx - pad); ry = max(0, ry - pad)
            rw_b = min(rw - rx, rw_b + 2*pad); rh_b = min(rh - ry, rh_b + 2*pad)
            best_bbox = [sx1 + rx, sy1 + ry, rw_b, rh_b]

    return best_bbox


def _color_fallback(img: np.ndarray, obj_type: str, obj_color: str,
                    max_objects: int, crop_offset: list = None) -> list:
    """Fallback global: cherche dans toute l'image."""
    h, w = img.shape[:2]
    offset_x = crop_offset[0] if crop_offset else 0
    offset_y = crop_offset[1] if crop_offset else 0

    mask = _get_color_mask(img, obj_type, obj_color)
    k7 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    margin = 15
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > img_area * 0.40: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= margin and y <= margin: continue
        if x + bw >= w - margin and y + bh >= h - margin: continue
        valid.append({"bbox": [x + offset_x, y + offset_y, bw, bh], "area": area})

    valid.sort(key=lambda c: c["area"], reverse=True)
    return [{"bbox": v["bbox"], "method": "color_fallback", "confidence": "medium"}
            for v in valid[:max_objects]]


def tool_object_localizer(image_input,
                           obj_type: str = "unknown",
                           obj_color: str = "unknown",
                           obj_shape: str = "unknown",
                           obj_size: str = "small",
                           location_text: str = "",
                           n_objects: int = 1,
                           crop_offset: list = None) -> dict:
    """
    v4: VLM donne x1%,y1%,x2%,y2% → conversion pixels → snap contour multi-stratégie.

    Args:
        crop_offset: [offset_x, offset_y] si l'image envoyée au localizer est un crop
                     → recalibrer les coords vers l'image complète

    Returns:
        found, bbox=[x,y,w,h], all_bboxes, method, confidence
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input.copy()

    if img is None or img.size == 0:
        return {"found": False, "bbox": None, "all_bboxes": [], "method": "error"}

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    crop_offset = crop_offset or [0, 0]
    off_x, off_y = crop_offset

    print(f"  [Localizer v4] {obj_type} ({obj_color}) | {w}×{h}px | n={n_objects}")

    # Encode pour le VLM
    b64, mtype = _encode(img)

    # VLM → bbox directement en %
    vlm_objects = _call_vlm_bbox(b64, mtype, obj_type, obj_color, obj_shape, n_objects)
    print(f"  [Localizer v4] VLM returned {len(vlm_objects)} bbox(es)")

    all_precise = []

    if vlm_objects:
        # Resize factor (si l'image a été réduite pour le VLM)
        vlm_h = min(h, 1024) if max(h, w) > 1024 else h
        vlm_w = min(w, 1024) if max(h, w) > 1024 else w
        scale_x = w / vlm_w if vlm_w != w else 1.0
        scale_y = h / vlm_h if vlm_h != h else 1.0

        for i, obj in enumerate(vlm_objects):
            conf = obj.get("confidence", "low")
            if conf == "low":
                print(f"    VLM obj {i+1}: low confidence → skipped")
                continue

            # Convertir % → pixels
            x1p = float(obj.get("x1_pct", 0))
            y1p = float(obj.get("y1_pct", 0))
            x2p = float(obj.get("x2_pct", 100))
            y2p = float(obj.get("y2_pct", 100))

            vx1, vy1, vx2, vy2 = _pct_to_pixels(x1p, y1p, x2p, y2p, w, h)
            print(f"    VLM obj {i+1}: pct=({x1p:.1f},{y1p:.1f},{x2p:.1f},{y2p:.1f}) → pixels=({vx1},{vy1},{vx2},{vy2})")

            # Zone de recherche élargie 2.5× autour du VLM bbox
            bw_v = max(vx2 - vx1, 10)
            bh_v = max(vy2 - vy1, 10)
            margin_x = int(bw_v * 0.8)
            margin_y = int(bh_v * 0.8)

            sx1 = max(0, vx1 - margin_x)
            sy1 = max(0, vy1 - margin_y)
            sx2 = min(w, vx2 + margin_x)
            sy2 = min(h, vy2 + margin_y)

            precise_bbox = _snap_to_best_contour(img, sx1, sy1, sx2, sy2, obj_type, obj_color)

            if precise_bbox:
                # Recalibrer vers image complète (en ajoutant l'offset du crop)
                px, py, pw, ph = precise_bbox
                global_bbox = [px + off_x, py + off_y, pw, ph]
                print(f"    → Snap OK: local={precise_bbox} global={global_bbox}")
                all_precise.append({"bbox": global_bbox, "method": "vlm_bbox+snap", "confidence": conf})
            else:
                # Fallback: utiliser le bbox VLM direct
                fallback = [vx1 + off_x, vy1 + off_y, vx2 - vx1, vy2 - vy1]
                print(f"    → Snap failed, VLM direct: {fallback}")
                all_precise.append({"bbox": fallback, "method": "vlm_bbox", "confidence": "medium"})

    if all_precise:
        primary  = all_precise[0]["bbox"]
        all_bbs  = [p["bbox"] for p in all_precise]
        method   = all_precise[0]["method"]
        print(f"  [Localizer v4] OK: {len(all_precise)} bbox(es)")
        for i, b in enumerate(all_bbs):
            print(f"    Object {i+1}: {b}")
        return {"found": True, "bbox": primary, "all_bboxes": all_bbs,
                "method": method, "confidence": all_precise[0]["confidence"]}

    # Fallback global couleur
    print(f"  [Localizer v4] No VLM bbox → color fallback")
    color_res = _color_fallback(img, obj_type, obj_color, n_objects + 1, crop_offset)
    if color_res:
        primary  = color_res[0]["bbox"]
        all_bbs  = [r["bbox"] for r in color_res]
        print(f"  [Localizer v4] Color fallback: {len(all_bbs)} object(s)")
        return {"found": True, "bbox": primary, "all_bboxes": all_bbs,
                "method": "color_fallback", "confidence": "medium"}

    # Dernier recours: approximation textuelle
    if location_text:
        t = location_text.lower()
        cx, cy = 0.5, 0.5
        wf, hf = 0.15, 0.10
        if "left" in t and "center" not in t: cx = 0.20
        elif "right" in t and "center" not in t: cx = 0.80
        if "top" in t and "center" not in t: cy = 0.20
        elif "bottom" in t and "center" not in t: cy = 0.80
        if "large" in t: wf, hf = 0.25, 0.20

        x = max(0, int((cx - wf/2) * w)) + off_x
        y = max(0, int((cy - hf/2) * h)) + off_y
        bw = int(wf * w); bh = int(hf * h)
        bbox = [x, y, min(bw, w), min(bh, h)]
        return {"found": True, "bbox": bbox, "all_bboxes": [bbox],
                "method": "text_approx", "confidence": "low"}

    return {"found": False, "bbox": None, "all_bboxes": [], "method": "failed", "confidence": "low"}