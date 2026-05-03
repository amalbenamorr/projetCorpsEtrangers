"""
tool_alert_commander.py — v3 FINAL
────────────────────────────────────
FIXES v3:
- Generates heatmap FROM bbox coordinates when no scanner heatmap available
  → GIF now always has a heatmap frame, even in VLM-first architecture
- Draws ALL detected bboxes (vlm_objects list), not just one
- GIF is always generated for REJECT/HUMAN_REVIEW
- Better label rendering with background box for readability
"""

import json
import time
from datetime import datetime
from pathlib import Path
from tools.tool_whatsapp_notifier import tool_whatsapp_notifier


import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

OUTPUTS_DIR = Path("outputs")
LOGS_DIR    = Path("logs")
OUTPUTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
ALERT_LOG = LOGS_DIR / "alerts.log"

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
BLINK  = "\033[5m"


def _build_heatmap_from_bboxes(img_shape: tuple,
                                 bboxes: list,
                                 scanner_heatmap=None) -> np.ndarray:
    """
    Build a heatmap image from detected bounding boxes.
    If scanner_heatmap is available, use it.
    Otherwise, synthesize one from the bbox positions — Gaussian blobs centered on each bbox.
    This ensures the GIF always has a meaningful heatmap frame.
    """
    h, w = img_shape[:2]

    if scanner_heatmap is not None and isinstance(scanner_heatmap, np.ndarray):
        hm = scanner_heatmap.astype(np.float32)
        # Normalize
        mn, mx = hm.min(), hm.max()
        if mx > mn:
            hm = (hm - mn) / (mx - mn)
        return hm

    # Synthesize heatmap from bboxes
    heatmap = np.zeros((h, w), dtype=np.float32)

    for bbox in bboxes:
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox]
        cx = x + bw // 2
        cy = y + bh // 2

        # Gaussian blob centered on the object
        sigma_x = max(bw, 40)
        sigma_y = max(bh, 40)

        Y, X = np.ogrid[:h, :w]
        gauss = np.exp(
            -((X - cx) ** 2 / (2 * sigma_x ** 2) +
              (Y - cy) ** 2 / (2 * sigma_y ** 2))
        )
        heatmap = np.maximum(heatmap, gauss.astype(np.float32))

    # If no bbox, uniform low heatmap
    if heatmap.max() < 1e-6:
        heatmap = np.ones((h, w), dtype=np.float32) * 0.1

    return heatmap


def _draw_all_detections(img: np.ndarray,
                          detected_bbox: list,
                          vlm_objects: list,
                          obj_type: str) -> np.ndarray:
    """
    Draw bounding boxes for ALL detected objects.
    - detected_bbox: primary localized bbox [x,y,w,h]
    - vlm_objects: list of all VLM-detected objects (may have multiple)
    """
    out = img.copy()
    h_img, w_img = out.shape[:2]

    drawn = []

    # Draw primary detected_bbox (most precise, from localizer)
    if detected_bbox and len(detected_bbox) == 4:
        x, y, bw, bh = [int(v) for v in detected_bbox]
        # Outer yellow border
        cv2.rectangle(out, (x-3, y-3), (x+bw+3, y+bh+3), (0, 255, 255), 2)
        # Inner red box
        cv2.rectangle(out, (x, y), (x+bw, y+bh), (0, 0, 255), 3)
        # Label with background
        label = f"DETECTED: {obj_type.upper()}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(y - 5, lh + 8)
        cv2.rectangle(out, (x, label_y - lh - 6), (x + lw + 6, label_y + 2), (0, 0, 180), -1)
        cv2.putText(out, label, (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        drawn.append((x, y, bw, bh))

    # If VLM found multiple objects, try to draw secondary ones too
    # (approximated from their location_description)
    if vlm_objects and len(vlm_objects) > 1:
        for i, obj in enumerate(vlm_objects[1:], 1):
            loc = (obj.get("location_description", "") or "").lower()
            # Try to infer approximate position from text
            cx_frac, cy_frac = 0.5, 0.5
            if "left" in loc:   cx_frac = 0.25
            if "right" in loc:  cx_frac = 0.75
            if "top" in loc:    cy_frac = 0.25
            if "bottom" in loc: cy_frac = 0.75

            size = obj.get("size", "small")
            w_frac = 0.12 if size == "small" else 0.20
            h_frac = 0.08 if size == "small" else 0.14

            ox = int((cx_frac - w_frac/2) * w_img)
            oy = int((cy_frac - h_frac/2) * h_img)
            obw = int(w_frac * w_img)
            obh = int(h_frac * h_img)

            # Avoid drawing on top of primary bbox
            overlaps = any(
                abs(ox - dx) < 50 and abs(oy - dy) < 50
                for dx, dy, _, _ in drawn
            )
            if not overlaps:
                cv2.rectangle(out, (ox, oy), (ox+obw, oy+obh), (0, 165, 255), 2)  # orange
                label2 = f"#{i+1}: {obj.get('type', 'unknown')}"
                cv2.putText(out, label2, (ox, max(oy-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                drawn.append((ox, oy, obw, obh))

    return out


def _generate_gif(original_img: np.ndarray,
                  heatmap,
                  zones: list,
                  output_path: Path,
                  detected_bbox: list = None,
                  vlm_objects: list = None,
                  obj_type: str = "foreign object") -> bool:
    """
    Generate 3-frame animated GIF:
    Frame 1: original image with precise bbox(es)
    Frame 2: heatmap overlay (always generated, even without scanner)
    Frame 3: blended
    """
    if not PIL_AVAILABLE or original_img is None:
        return False

    try:
        h, w = original_img.shape[:2]
        target_size = (640, 480)
        vlm_objects = vlm_objects or []

        # Collect all bboxes for heatmap generation
        all_bboxes = []
        if detected_bbox and len(detected_bbox) == 4:
            all_bboxes.append(detected_bbox)
        # Also include scanner zones as secondary
        for zone in zones[:3]:
            if "bbox" in zone:
                all_bboxes.append(zone["bbox"])

        # ── Frame 1: original + precise bbox annotations ──────────────────
        f1 = _draw_all_detections(original_img, detected_bbox, vlm_objects, obj_type)
        f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f1_pil = PILImage.fromarray(f1_rgb).resize(target_size)

        # ── Frame 2: heatmap overlay ──────────────────────────────────────
        hm = _build_heatmap_from_bboxes(original_img.shape, all_bboxes, heatmap)

        # Resize heatmap to match image
        if hm.shape[:2] != (h, w):
            hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)

        hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        f2 = cv2.addWeighted(original_img, 0.40, hm_color, 0.60, 0)

        # Draw bbox on heatmap frame too
        if detected_bbox and len(detected_bbox) == 4:
            x, y, bw, bh = [int(v) for v in detected_bbox]
            cv2.rectangle(f2, (x, y), (x+bw, y+bh), (255, 255, 255), 3)
            cv2.putText(f2, obj_type.upper(), (x, max(y-8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        f2_pil = PILImage.fromarray(f2_rgb).resize(target_size)

        # ── Frame 3: blended ──────────────────────────────────────────────
        f3 = cv2.addWeighted(f1, 0.65, f2, 0.35, 0)
        f3_rgb = cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)
        f3_pil = PILImage.fromarray(f3_rgb).resize(target_size)

        # Save animated GIF
        f1_pil.save(
            str(output_path),
            save_all=True,
            append_images=[f2_pil, f3_pil],
            loop=0,
            duration=500
        )
        return True

    except Exception as e:
        print(f"[GIF ERROR] {e}")
        import traceback; traceback.print_exc()
        return False


def _generate_tts(message: str, output_path: Path) -> bool:
    if not TTS_AVAILABLE:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(message, str(output_path))
        engine.runAndWait()
        return True
    except Exception:
        return False


def _print_terminal(decision: str, report: dict):
    now     = datetime.now().strftime("%H:%M:%S")
    obj     = report.get("object_detected", "unknown")
    hazard  = report.get("hazard_level", "unknown")
    score   = report.get("anomaly_score", "?")
    material= report.get("dominant_material", "unknown")
    station = report.get("station", "unknown")
    bbox    = report.get("detected_bbox")

    print("\n" + "═" * 60)
    if decision == "REJECT":
        print(f"{BOLD}{RED}{BLINK}🚨  LOT REJECTED — {now}{RESET}")
        print(f"{RED}Object   : {obj}{RESET}")
        print(f"{RED}Material : {material}{RESET}")
        print(f"{RED}Hazard   : {hazard}{RESET}")
        print(f"{RED}Score    : {score}{RESET}")
        print(f"{RED}Station  : {station}{RESET}")
        if bbox:
            print(f"{RED}Location : bbox={bbox}{RESET}")
    elif decision == "HUMAN_REVIEW":
        print(f"{BOLD}{YELLOW}⚠️   HUMAN REVIEW REQUIRED — {now}{RESET}")
        print(f"{YELLOW}Object   : {obj}{RESET}")
        print(f"{YELLOW}Hazard   : {hazard}{RESET}")
        print(f"{YELLOW}Score    : {score}{RESET}")
        if bbox:
            print(f"{YELLOW}Location : bbox={bbox}{RESET}")
    else:
        print(f"{BOLD}{GREEN}✅  ACCEPTED — {now}{RESET}")
        print(f"{GREEN}No foreign object detected.{RESET}")

    reasoning = report.get("brain_reasoning", "")
    if reasoning:
        print(f"\n{CYAN}Reasoning: {reasoning[:200]}{RESET}")
    print("═" * 60 + "\n")


def tool_alert_commander(
    decision: str,
    report: dict,
    full_image: np.ndarray = None,
    heatmap=None,
    zones: list = None,
    detected_bbox: list = None
) -> dict:
    """
    Execute the brain's final decision.

    Args:
        decision      : ACCEPT | REJECT | HUMAN_REVIEW
        report        : full case data
        full_image    : numpy BGR image
        heatmap       : float32 heatmap from scanner (can be None — will be synthesized from bbox)
        zones         : scanner zones (optional, for fallback bbox on GIF)
        detected_bbox : [x, y, w, h] primary localized object bbox
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    zones      = zones or []
    obj_type   = report.get("object_detected", "foreign object")
    vlm_objects = report.get("vlm_objects", [])

    # Log entry
    base_name   = f"{decision.lower()}_{timestamp}"
    gif_path    = OUTPUTS_DIR / f"{base_name}.gif"
    report_path = OUTPUTS_DIR / f"{base_name}_report.json"

    log_entry = {
        "timestamp":     datetime.now().isoformat(),
        "decision":      decision,
        "object":        report.get("object_detected", "unknown"),
        "material":      report.get("dominant_material", "unknown"),
        "hazard":        report.get("hazard_level", "unknown"),
        "anomaly_score": report.get("anomaly_score"),
        "station":       report.get("station", "unknown"),
        "root_cause":    report.get("root_cause", "unknown"),
        "detected_bbox": detected_bbox,
        "gif_path":      str(gif_path) if decision != "ACCEPT" else None,
        "report_path":   str(report_path) if decision != "ACCEPT" else None
    }
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    _print_terminal(decision, {**report, "detected_bbox": detected_bbox})

    if decision == "ACCEPT":
        return {
            "executed":    True,
            "gif_path":    None,
            "audio_path":  None,
            "log_path":    str(ALERT_LOG),
            "report_path": None,
            "verdict":     "ACCEPT logged silently"
        }

    # REJECT / HUMAN_REVIEW → generate GIF + report
    audio_path  = OUTPUTS_DIR / f"{base_name}.mp3"

    gif_generated   = False
    audio_generated = False

    if full_image is not None:
        gif_generated = _generate_gif(
            original_img=full_image,
            heatmap=heatmap,           # can be None → will synthesize from bbox
            zones=zones,
            output_path=gif_path,
            detected_bbox=detected_bbox,
            vlm_objects=vlm_objects,
            obj_type=obj_type
        )

    tts_text = f"Alert. {decision}. {obj_type} detected at {report.get('station','unknown')}. Immediate action required."
    audio_generated = _generate_tts(tts_text, audio_path)

    full_report = {**log_entry, **report, "tts_message": tts_text, "detected_bbox": detected_bbox}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    
    

    return {
        "executed":        True,
        "gif_path":        str(gif_path) if gif_generated else None,
        "audio_path":      str(audio_path) if audio_generated else None,
        "log_path":        str(ALERT_LOG),
        "report_path":     str(report_path),
        "gif_generated":   gif_generated,
        "audio_generated": audio_generated,
        "detected_bbox":   detected_bbox,
        "verdict":         f"{decision} — gif={gif_generated} bbox={'yes' if detected_bbox else 'no'}"
    }