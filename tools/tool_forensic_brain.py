"""
tool_forensic_brain.py — v7 FINAL
────────────────────────────────────
FIXES v7 over v6:
- Passes white_score and texture_score from the scanner zone into VLM context
  → VLM receives: "scanner white_score=0.8 (feather/bone likely)" as hint
- Feather-specific prompt: if white_score high → prompt emphasizes feather check
- wood vs feather disambiguation in prompt when brown_ratio low + white high
- Still uses llama-4-scout as primary VLM
- Still runs up to 3 calls with majority vote
"""

import base64
import re
import json
import requests
import time
from pathlib import Path
from PIL import Image
import numpy as np
import io
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

FACULTY_API_BASE = os.getenv("FACULTY_API_BASE", "http://your-faculty-api/v1")
FACULTY_API_KEY  = os.getenv("FACULTY_API_KEY", "")
VLM_MODEL        = os.getenv("FACULTY_VLM_MODEL", "hosted_vllm/llava-1.5-7b-hf")

GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_VLM_MODEL = os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

VLM_MAX_DIM = 1024
ZOOM_SIZE   = 320


def _annotate_full_image(full_image: np.ndarray, bbox: list) -> np.ndarray:
    annotated = full_image.copy()
    h, w = annotated.shape[:2]

    scale = 1.0
    if max(h, w) > VLM_MAX_DIM:
        scale = VLM_MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x, y, bw, bh = bbox
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + bw) * scale)
    y2 = int((y + bh) * scale)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label_y = max(y1 - 12, 20)
    cv2.putText(annotated, ">>> INSPECT THIS ZONE <<<", (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return annotated


def _extract_zoom_crop(full_image: np.ndarray, bbox: list, size: int = ZOOM_SIZE) -> np.ndarray:
    h_img, w_img = full_image.shape[:2]
    x, y, bw, bh = bbox

    margin_x = max(int(bw * 0.6), 30)
    margin_y = max(int(bh * 0.6), 30)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + bw + margin_x)
    y2 = min(h_img, y + bh + margin_y)

    crop = full_image[y1:y2, x1:x2].copy()

    bx1 = x - x1
    by1 = y - y1
    bx2 = bx1 + bw
    by2 = by1 + bh
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

    ch, cw = crop.shape[:2]
    if ch < size or cw < size:
        scale = max(size / max(ch, 1), size / max(cw, 1))
        new_h, new_w = int(ch * scale), int(cw * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop


def _encode_image(image_array: np.ndarray) -> tuple:
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def _build_prompt(context: dict, attempt: int = 1, zoom_available: bool = False) -> str:
    hints = []

    # Anomaly score hint
    if context.get("anomaly_score"):
        thr = context.get("threshold_used", 0.25)
        hints.append(f"anomaly score={context['anomaly_score']:.3f} (threshold={thr:.3f})")

    # Spectral material hints
    dominant = context.get("dominant_material", "")
    if dominant and dominant not in ("organic",):
        hints.append(f"spectral dominant={dominant}")

    # Color profile hints — CRITICAL for feather vs wood disambiguation
    cp = context.get("color_profile", {})
    if cp:
        if cp.get("white_ratio", 0) > 0.10:
            hints.append(f"COLOR: HIGH white_ratio={cp['white_ratio']:.2f} → likely feather/bone/plastic (NOT wood)")
        if cp.get("brown_ratio", 0) > 0.30:
            hints.append(f"COLOR: HIGH brown_ratio={cp['brown_ratio']:.2f} → likely wood/organic material")
        if cp.get("gray_ratio", 0) > 0.20:
            hints.append(f"COLOR: HIGH gray_ratio={cp['gray_ratio']:.2f} → likely metal/bone")

    # Scanner zone signals
    white_score = context.get("white_score", 0)
    texture_score = context.get("texture_score", 0)
    if white_score > 0.40:
        hints.append(f"SCANNER: white_score={white_score:.2f} → strong white/bright object signal → LOOK FOR FEATHER OR BONE")
    if texture_score > 0.60:
        hints.append(f"SCANNER: texture_score={texture_score:.2f} → strong texture anomaly detected")

    context_str = f"\nSensor evidence:\n" + "\n".join(f"  - {h}" for h in hints) if hints else ""
    zoom_note = "\nA ZOOMED CROP of the red rectangle is also provided as the second image." if zoom_available else ""

    extra = ""
    if attempt == 2:
        extra = "\n\nSECOND INSPECTION — Be extremely critical. Re-examine color and shape carefully."
    elif attempt == 3:
        extra = "\n\nFINAL TIE-BREAKER — Maximum scrutiny. Focus on: is there ANY non-meat material?"

    # Build feather-specific alert if white signal is strong
    feather_alert = ""
    if white_score > 0.35 or (cp and cp.get("white_ratio", 0) > 0.10):
        feather_alert = """
⚠️  FEATHER ALERT: sensors detected a white/bright object.
Feathers look like: white or light gray, thin elongated filaments, soft pointed tip, 
barbs visible along a central rachis, lying flat on/in the meat.
Check carefully for this pattern inside the red rectangle."""

    return f"""You are a FORENSIC INSPECTOR in a poultry/meat processing plant.
Identify foreign objects in meat. Examine ONLY what is inside the RED RECTANGLE.{context_str}{zoom_note}{feather_alert}{extra}

FOREIGN OBJECTS — what they look like:
- metal_fragment: dark gray/silver, hard sharp edges, flat or irregular, reflective
- bone_shard: white/ivory, rigid, smooth or jagged edges, opaque, dense
- wood: brown/dark brown, cylindrical or splinter, fibrous grain texture, rough
- plastic: any solid color, uniform smooth surface, may be translucent
- feather: WHITE or light gray, thin elongated shape, soft filamentous barbs, central rachis visible
- glass: transparent/translucent, smooth, may reflect rainbow colors
- rubber: black/dark, elastic-looking, smooth matte surface
- stone: gray/brown, rough irregular surface, matte

CLEAN MEAT: pink/red uniform color, organic fibrous texture, no hard edges, continuous surface.
CHICKEN SKIN/FAT: pale pink/cream, smooth, organic — do NOT flag as foreign object.

Respond ONLY with valid JSON (no markdown):
{{
  "description": "<precise visual description of what is inside the red rectangle>",
  "object_detected": "<metal_fragment | bone_shard | wood | plastic | feather | glass | rubber | stone | clean_meat | unknown>",
  "shape": "<rectangular | cylindrical | splinter | elongated | irregular | flat | filamentous | etc>",
  "color": "<exact color: dark gray | silver | white | ivory | brown | translucent | black | pale pink | etc>",
  "surface_texture": "<shiny | matte | rough | smooth | fibrous | grainy | filamentous | reflective>",
  "hazard_level": "<critical | moderate | low | none>",
  "confidence": "<high | medium | low>",
  "reasoning": "<visual evidence: color vs surrounding meat, edge type, texture, shape — be specific>"
}}"""


def _call_groq_vlm(b64: str, media_type: str, prompt: str, b64_zoom: str = None) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }
    content = [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}]
    if b64_zoom:
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_zoom}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": GROQ_VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 700,
        "temperature": 0.05
    }

    for attempt in range(5):
        try:
            r = requests.post(f"{GROQ_API_BASE}/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = min((2 ** attempt) * 5, 45)
                print(f"  [⏳ VLM 429] Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3)
    raise RuntimeError("Groq VLM failed after retries")


def _call_faculty_vlm(b64: str, media_type: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {FACULTY_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 700,
        "temperature": 0.05
    }
    r = requests.post(f"{FACULTY_API_BASE}/chat/completions",
                      headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_vlm_response(raw: str) -> dict:
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}


def _normalize_object(obj: str) -> str:
    foreign_keywords = [
        "metal", "bone", "plastic", "feather", "glass", "rubber",
        "wood", "stone", "wire", "blade", "fragment", "shard",
        "screw", "nail", "clip", "tag", "ring", "splinter", "stick"
    ]
    obj_lower = (obj or "").lower()
    for kw in foreign_keywords:
        if kw in obj_lower:
            return obj if obj else kw
    if "clean" in obj_lower or "meat" in obj_lower or "skin" in obj_lower or "fat" in obj_lower:
        return "clean_meat"
    return "unknown"


def _majority_vote(results: list) -> dict:
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    objects = [_normalize_object(r.get("object_detected", "unknown")) for r in results]
    from collections import Counter
    obj_counts = Counter(objects)
    most_common_obj, count = obj_counts.most_common(1)[0]

    if count >= 2 and most_common_obj != "unknown":
        conf_order = {"high": 3, "medium": 2, "low": 1}
        best = max(
            [r for r in results if _normalize_object(r.get("object_detected", "")) == most_common_obj],
            key=lambda r: conf_order.get(r.get("confidence", "low"), 0)
        )
        best = dict(best)
        if count >= 2 and best.get("confidence") != "high":
            best["confidence"] = "high"
            best["reasoning"] = f"[CONFIRMED {count}/{len(results)} VLM calls] " + best.get("reasoning", "")
        return best

    conf_order = {"high": 3, "medium": 2, "low": 1}
    non_unknown = [r for r in results if _normalize_object(r.get("object_detected", "")) != "unknown"]
    if non_unknown:
        return max(non_unknown, key=lambda r: conf_order.get(r.get("confidence", "low"), 0))
    return results[0]


def tool_forensic_brain(image_input,
                        full_image: np.ndarray = None,
                        bbox: list = None,
                        context: dict = None) -> dict:
    """
    v7: Full image + zoom crop → llama-4-scout (primary VLM).
    Context now includes white_score + texture_score from scanner for feather detection.
    Up to 3 VLM calls with majority vote.
    """
    context = context or {}

    # Build annotated images
    zoom_crop = None
    if full_image is not None and isinstance(full_image, np.ndarray) and bbox is not None:
        annotated = _annotate_full_image(full_image, bbox)
        zoom_crop = _extract_zoom_crop(full_image, bbox)
        print(f"  [🔬 VLM v7] Full image: {annotated.shape[1]}×{annotated.shape[0]}px | "
              f"bbox={bbox} | zoom={zoom_crop.shape[1]}×{zoom_crop.shape[0]}px")
    elif isinstance(image_input, np.ndarray):
        annotated = image_input.copy()
        h, w = annotated.shape[:2]
        margin = max(5, min(w, h) // 8)
        cv2.rectangle(annotated, (margin, margin), (w - margin, h - margin), (0, 0, 255), 3)
        cv2.putText(annotated, ">>> INSPECT THIS ZONE <<<", (margin, max(margin - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"  [🔬 VLM v7] Patch fallback: {annotated.shape[1]}×{annotated.shape[0]}px")
    else:
        annotated = image_input

    if isinstance(annotated, np.ndarray):
        b64_full, media_type = _encode_image(annotated)
    else:
        b64_full, media_type = _encode_path(annotated)

    b64_zoom = None
    if zoom_crop is not None and isinstance(zoom_crop, np.ndarray):
        b64_zoom, _ = _encode_image(zoom_crop)

    # First VLM call
    prompt1 = _build_prompt(context, attempt=1, zoom_available=(b64_zoom is not None))
    raw1 = ""
    model_used = "none"
    vlm_error = None

    for fn, name, kwargs in [
        (_call_groq_vlm, GROQ_VLM_MODEL, {"b64_zoom": b64_zoom}),
        (_call_faculty_vlm, VLM_MODEL, {})
    ]:
        try:
            if fn == _call_groq_vlm:
                raw1 = fn(b64_full, media_type, prompt1, **kwargs)
            else:
                raw1 = fn(b64_full, media_type, prompt1)
            model_used = name
            break
        except Exception as e:
            vlm_error = str(e)
            print(f"  [🔬 VLM] {name} failed: {e}")
            continue

    if not raw1:
        return {
            "description": "VLM unavailable",
            "object_detected": "unknown",
            "shape": "unknown", "color": "unknown", "surface_texture": "unknown",
            "hazard_level": "unknown", "confidence": "low",
            "raw_vlm_output": "", "model_used": "none",
            "error": vlm_error,
            "verdict": "vlm_failed — rely on anomaly_hunter + spectral_eye"
        }

    parsed1 = _parse_vlm_response(raw1)
    confidence1 = parsed1.get("confidence", "low")
    object1 = _normalize_object(parsed1.get("object_detected", "unknown"))

    print(f"  [🔬 VLM v7] Call 1: object={object1} conf={confidence1}")

    all_results = [parsed1]

    # Multi-call voting when not high confidence
    if confidence1 != "high" or object1 == "unknown":
        for attempt_n in [2, 3]:
            try:
                prompt_n = _build_prompt(context, attempt=attempt_n, zoom_available=(b64_zoom is not None))
                raw_n = _call_groq_vlm(b64_full, media_type, prompt_n, b64_zoom=b64_zoom)
                parsed_n = _parse_vlm_response(raw_n)
                obj_n = _normalize_object(parsed_n.get("object_detected", "unknown"))
                conf_n = parsed_n.get("confidence", "low")
                print(f"  [🔬 VLM v7] Call {attempt_n}: object={obj_n} conf={conf_n}")
                all_results.append(parsed_n)

                if len(all_results) >= 2:
                    objs_so_far = [_normalize_object(r.get("object_detected", "unknown")) for r in all_results]
                    if objs_so_far.count(obj_n) >= 2 and obj_n != "unknown" and conf_n == "high":
                        print(f"  [🔬 VLM v7] Early consensus: {obj_n} — stopping")
                        break
            except Exception as e:
                print(f"  [🔬 VLM v7] Call {attempt_n} failed: {e}")
                break

    final_parsed = _majority_vote(all_results)
    final_obj = _normalize_object(final_parsed.get('object_detected', 'unknown'))
    final_conf = final_parsed.get('confidence', 'low')
    print(f"  [🔬 VLM v7] FINAL ({len(all_results)} calls): object={final_obj} conf={final_conf}")

    description     = final_parsed.get("description", raw1[:300])
    object_detected = _normalize_object(final_parsed.get("object_detected", "unknown"))
    shape           = final_parsed.get("shape", "unknown")
    color           = final_parsed.get("color", "unknown")
    surface_texture = final_parsed.get("surface_texture", "unknown")
    hazard_level    = final_parsed.get("hazard_level", "unknown")
    confidence      = final_parsed.get("confidence", "low")
    reasoning       = final_parsed.get("reasoning", "")

    verdict = (
        f"object={object_detected} hazard={hazard_level} "
        f"confidence={confidence} calls={len(all_results)} "
        f"model={model_used.split('/')[-1]}"
    )

    return {
        "description":     description,
        "object_detected": object_detected,
        "shape":           shape,
        "color":           color,
        "surface_texture": surface_texture,
        "hazard_level":    hazard_level,
        "confidence":      confidence,
        "reasoning":       reasoning,
        "raw_vlm_output":  raw1[:500],
        "model_used":      model_used,
        "n_vlm_calls":     len(all_results),
        "verdict":         verdict
    }


def _encode_path(image_path) -> tuple:
    with open(image_path, "rb") as f:
        raw = f.read()
    ext = str(image_path).lower().split(".")[-1]
    media_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return base64.b64encode(raw).decode("utf-8"), media_type