"""
tool_direct_vlm_inspector.py — v2 FINAL
─────────────────────────────────────────
FIXES v2:
- Prompt explicitement: convoyeur/rails/surfaces métalliques au bord = IGNORER
- Crop central avant envoi: supprime 8% de bordure (convoyeur toujours en bord)
- Focus exclusif: poultry/meat products uniquement
- Si objet détecté EST dans la zone de bordure → downgrade confidence
"""

import base64, re, json, requests, time, cv2, numpy as np, io, os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_VLM_MODEL = os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MAX_DIM       = 1280
BORDER_MARGIN = 0.08  # 8% de bordure ignorée (convoyeur)


def _encode(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _crop_central(img: np.ndarray, margin: float = BORDER_MARGIN) -> tuple:
    """
    Retourne l'image croppée sans les bordures (convoyeur).
    Aussi retourne les offsets pour recalculer les coords globales.
    """
    h, w = img.shape[:2]
    x1 = int(w * margin)
    y1 = int(h * margin)
    x2 = int(w * (1 - margin))
    y2 = int(h * (1 - margin))
    return img[y1:y2, x1:x2].copy(), x1, y1


def _call_vlm(b64: str, media_type: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 900, "temperature": 0.05
    }
    for attempt in range(5):
        try:
            r = requests.post(f"{GROQ_API_BASE}/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = min((2**attempt)*5, 45)
                print(f"  [VLM 429] wait {wait}s"); time.sleep(wait); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 4: raise
            time.sleep(3)
    raise RuntimeError("VLM failed")


PROMPT_FULL_IMAGE = """You are an expert food safety inspector in a POULTRY and MEAT processing plant.

YOUR ONLY JOB: find foreign objects contaminating the MEAT/POULTRY PRODUCT itself.

━━━ WHAT TO COMPLETELY IGNORE (these are normal, DO NOT flag them) ━━━
- Conveyor belt surface (blue, gray, metallic grid — always at image edges/borders)
- Metal conveyor rails, tracks, rollers (at image edges)
- Processing equipment parts visible at image borders
- Shadows from overhead lighting
- Ice crystals or water condensation on meat surface
- White chicken fat/skin (pale cream/pink, soft organic texture — very common)
- Natural bone at visible joints (thighs, drumsticks — expected)
- Pink/red meat color variations (normal muscle tissue)

━━━ WHAT COUNTS AS A FOREIGN OBJECT (contamination IN the meat) ━━━
ONLY flag objects that are INSIDE/ON the meat product, NOT at image edges:
- Metal fragment: dark gray/silver, hard sharp irregular edges, ON THE MEAT SURFACE
- Wood splinter: brown, cylindrical/fibrous grain, embedded IN or ON the meat
- Plastic: any color, uniform smooth edges, rigid piece ON the meat
- Feather: white/gray, thin elongated soft filaments with central rachis, ON the meat
- Bone shard: white/ivory, unexpected sharp fragments NOT at natural joints
- Glass, rubber, stone: hard foreign material ON/IN the meat

━━━ IMPORTANT RULE ━━━
If a suspicious object is ONLY visible at the image border/edge (within 10% of image edge),
it is almost certainly part of the conveyor equipment. Set confidence=low for those.
Focus on objects clearly ON the meat product itself.

Scan this image carefully. Respond ONLY with valid JSON:
{
  "foreign_objects_found": true or false,
  "n_objects": <number>,
  "objects": [
    {
      "type": "<metal_fragment|bone_shard|wood|plastic|feather|glass|rubber|stone>",
      "location_description": "<where on the MEAT: center, top-left of meat piece, etc.>",
      "color": "<exact color>",
      "shape": "<shape>",
      "size": "<small|medium|large>",
      "confidence": "<high|medium|low>",
      "hazard_level": "<critical|moderate|low>",
      "is_on_meat": true or false,
      "reasoning": "<visual evidence distinguishing it from normal meat/fat/conveyor>"
    }
  ],
  "overall_verdict": "<ACCEPT|REJECT|HUMAN_REVIEW>",
  "verdict_confidence": "<high|medium|low>",
  "general_description": "<brief description of what you see>"
}

If no foreign objects on the meat: foreign_objects_found=false, n_objects=0, objects=[], overall_verdict=ACCEPT"""


def tool_direct_vlm_inspector(image_input) -> dict:
    """
    v2: Crop border before sending to VLM → eliminates conveyor false positives.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return {"error": "cannot load image", "overall_verdict": "ERROR"}
    else:
        img = image_input.copy()

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h_orig, w_orig = img.shape[:2]
    print(f"  [DirectVLM v2] Full image {w_orig}×{h_orig}px")

    # Crop central zone (remove conveyor border)
    img_central, offset_x, offset_y = _crop_central(img, BORDER_MARGIN)
    h_crop, w_crop = img_central.shape[:2]
    print(f"  [DirectVLM v2] Central crop {w_crop}×{h_crop}px (border {int(BORDER_MARGIN*100)}% removed)")

    b64, mtype = _encode(img_central)

    raw = ""
    try:
        raw = _call_vlm(b64, mtype, PROMPT_FULL_IMAGE)
        print(f"  [DirectVLM v2] Response: {len(raw)} chars")
    except Exception as e:
        print(f"  [DirectVLM v2] VLM failed: {e}")
        return {
            "foreign_objects_found": False, "n_objects": 0, "objects": [],
            "overall_verdict": "HUMAN_REVIEW", "verdict_confidence": "low",
            "error": str(e), "vlm_raw": "",
            "crop_offset": [offset_x, offset_y]
        }

    parsed = {}
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
    except Exception:
        parsed = {}

    found   = parsed.get("foreign_objects_found", False)
    objects = parsed.get("objects", [])
    n       = parsed.get("n_objects", len(objects))
    verdict = parsed.get("overall_verdict", "HUMAN_REVIEW")
    conf    = parsed.get("verdict_confidence", "low")

    # Filter: remove objects not on meat (is_on_meat=false)
    objects = [o for o in objects if o.get("is_on_meat", True)]
    if len(objects) == 0:
        found   = False
        n       = 0
        verdict = "ACCEPT"
        conf    = "high"

    # Downgrade: if REJECT with low confidence → HUMAN_REVIEW
    if verdict == "REJECT" and conf == "low":
        verdict = "HUMAN_REVIEW"

    # Downgrade: all objects have low confidence only
    if found and objects:
        good_conf = [o for o in objects if o.get("confidence") in ("high", "medium")]
        if not good_conf:
            verdict = "HUMAN_REVIEW"
            conf    = "low"
        else:
            objects = good_conf  # keep only reliable detections
            n = len(objects)

    print(f"  [DirectVLM v2] found={found} n={n} verdict={verdict} conf={conf}")
    for obj in objects:
        print(f"    → {obj.get('type')} ({obj.get('confidence')}) at {obj.get('location_description')}")

    return {
        "foreign_objects_found": found,
        "n_objects":             n,
        "objects":               objects,
        "overall_verdict":       verdict,
        "verdict_confidence":    conf,
        "general_description":   parsed.get("general_description", ""),
        "vlm_raw":               raw[:800],
        "image_shape":           [h_orig, w_orig],
        "crop_offset":           [offset_x, offset_y],  # pour recalcul coords globales
    }