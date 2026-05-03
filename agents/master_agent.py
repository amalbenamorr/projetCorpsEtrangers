"""
agents/master_agent.py — v12 FINAL
Scanner used only for heatmap GIF, not for detection.
crop_offset passed to localizer for correct bbox coordinates.
VLM-first architecture with PatchCore statistical confirmation.
Replace your agents/master_agent.py with this file.
"""

import json, os, re, time, traceback
from datetime import datetime
from pathlib import Path
import cv2, numpy as np, requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_direct_vlm_inspector import tool_direct_vlm_inspector
from tools.tool_object_localizer     import tool_object_localizer
from tools.tool_anomaly_hunter       import tool_anomaly_hunter
from tools.tool_spectral_eye         import tool_spectral_eye
from tools.tool_memory_search        import tool_memory_search
from tools.tool_web_investigator     import tool_web_investigator
from tools.tool_alert_commander      import tool_alert_commander
from tools.tool_memory_writer        import tool_memory_writer

try:
    from tools.tool_scanner import tool_scanner as _scanner_fn
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False

TOOL_REGISTRY = {
    "tool_direct_vlm_inspector": {
        "fn": tool_direct_vlm_inspector, "needs_image": True,
        "desc": (
            "PRIMARY — VLM inspects FULL IMAGE. Ignores conveyor/equipment at borders. "
            "Finds foreign objects contaminating the MEAT product itself. "
            "Returns: foreign_objects_found, n_objects, objects[{type,location,color,shape,size,confidence}], "
            "overall_verdict(ACCEPT|REJECT|HUMAN_REVIEW), verdict_confidence, crop_offset. "
            "ALWAYS CALL FIRST. Args: {}."
        )
    },
    "tool_object_localizer": {
        "fn": tool_object_localizer, "needs_image": True,
        "desc": (
            "LOCALIZE — finds precise pixel bbox for ALL detected objects. "
            "VLM gives x1%,y1%,x2%,y2% converted to pixels with contour snap. "
            "Returns: found, bbox=[x,y,w,h] (primary), all_bboxes. "
            "Call AFTER tool_direct_vlm_inspector when foreign objects found. "
            "Args: {obj_type, obj_color, obj_shape, obj_size, location_text, n_objects}."
        )
    },
    "tool_anomaly_hunter": {
        "fn": tool_anomaly_hunter, "needs_image": True,
        "desc": (
            "CONFIRM — PatchCore vs 4730 clean meat images. Uses detected_bbox crop. "
            "Returns: score, is_anomaly, confidence, threshold_used. Args: {}."
        )
    },
    "tool_spectral_eye": {
        "fn": tool_spectral_eye, "needs_image": True,
        "desc": (
            "MATERIAL — FFT+color analysis. Returns: dominant_material. "
            "Optional: call if VLM material uncertain. Args: {}."
        )
    },
    "tool_memory_search": {
        "fn": tool_memory_search, "needs_image": False,
        "desc": (
            "HISTORY — search past cases (similarity >= 0.75). "
            "Args: {context:{object_detected, dominant_material, shape, color}}."
        )
    },
    "tool_web_investigator": {
        "fn": tool_web_investigator, "needs_image": False,
        "desc": (
            "WEB — food-safety search. ONLY if object unknown AND memory gave no reliable match. "
            "Args: {context:{object_detected, dominant_material}}."
        )
    },
    "tool_alert_commander": {"fn": tool_alert_commander, "needs_image": False, "desc": "AUTO. Do not call."},
    "tool_memory_writer":   {"fn": tool_memory_writer,   "needs_image": False, "desc": "AUTO. Do not call."},
}

SYSTEM_PROMPT = """You are ELMAZRAA — forensic AI inspector for poultry/meat processing lines.
Detect foreign objects contaminating meat products. Output: ACCEPT / REJECT / HUMAN_REVIEW.

TOOLS:
{tool_descriptions}

## ReAct FORMAT — ONE action per response
THOUGHT: <reasoning with actual scores>
ACTION: <tool_name or FINAL_DECISION>
ARGS: <valid JSON — no image arrays>

## WORKFLOW (fully agentic)

Step 1: ALWAYS call tool_direct_vlm_inspector first.
  It ignores conveyor belts and equipment at image borders.
  It detects ONLY objects contaminating the meat product.

Step 2: Based on result:
  ACCEPT + high confidence → FINAL_DECISION(ACCEPT) immediately
  REJECT → call tool_object_localizer then tool_anomaly_hunter → FINAL_DECISION
  HUMAN_REVIEW → call tool_object_localizer + tool_anomaly_hunter → FINAL_DECISION

Optional (use intelligently, not always):
  tool_spectral_eye: if material ambiguous
  tool_memory_search: for root_cause
  tool_web_investigator: ONLY if truly unknown AND memory had no match

## DECISION TABLE
| VLM verdict | conf    | anomaly score   | Final        |
|-------------|---------|-----------------|--------------|
| ACCEPT      | high    | any             | ACCEPT       |
| ACCEPT      | med/low | < threshold     | ACCEPT       |
| REJECT      | high    | > threshold     | REJECT       |
| REJECT      | med     | > threshold     | REJECT       |
| REJECT      | any     | < threshold     | HUMAN_REVIEW |
| HUMAN_REVIEW| any     | > threshold*1.5 | HUMAN_REVIEW |
| HUMAN_REVIEW| any     | < threshold     | ACCEPT       |

## FINAL_DECISION format
ACTION: FINAL_DECISION
ARGS: {{
  "decision": "REJECT",
  "object_detected": "wood",
  "dominant_material": "wood",
  "hazard_level": "moderate",
  "root_cause": "wooden processing equipment fragment",
  "station": "Station-1",
  "brain_reasoning": "DirectVLM=REJECT conf=high. 2 wood splinters. score=0.32>thr=0.22.",
  "confidence": "high"
}}

Do NOT call tool_alert_commander or tool_memory_writer.
"""


def _format_descs():
    return "\n".join(f"  {n}: {i['desc']}" for n, i in TOOL_REGISTRY.items())


def _call_llm(messages):
    trimmed = _trim(messages, 6)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": trimmed,
               "max_tokens": 1000, "temperature": 0.05, "stop": ["OBSERVATION:"]}
    for attempt in range(6):
        try:
            r = requests.post(f"{GROQ_API_BASE}/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = min((2**attempt)*4, 60)
                print(f"[⏳ 429] {wait}s"); time.sleep(wait); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout: time.sleep(5)
        except Exception:
            if attempt == 5: raise
            time.sleep(3)
    raise RuntimeError("LLM unavailable")


def _trim(messages, keep=6):
    if len(messages) <= 3: return messages
    sys_msg, first = messages[0], messages[1]
    recent = messages[2:]; n = keep * 2
    if len(recent) > n:
        recent = [{"role": "user", "content": f"[{len(recent)-n} steps trimmed]"}] + recent[-n:]
    return [sys_msg, first] + recent


def _parse(text):
    m = re.search(r"ACTION:\s*(\w+)", text)
    if not m: return "", {}
    action = m.group(1).strip(); args = {}
    p = text.find("ARGS:", m.start())
    if p != -1:
        b = text.find("{", p)
        if b != -1:
            depth, end = 0, b
            for i, ch in enumerate(text[b:], b):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0: end = i; break
            raw = text[b:end+1]
            try: args = json.loads(raw)
            except:
                try: args = json.loads(re.sub(r"[\x00-\x1f\x7f]", "", raw))
                except: pass
    return action, args


def _fix_args(tool_name, args):
    if tool_name == "tool_memory_search" and "context" not in args:
        ctx_keys = {"object_detected","dominant_material","description","shape","color","hazard_level"}
        ext = {k: v for k, v in args.items() if k in ctx_keys}
        return {"context": ext} if ext else {"context": {"description": "poultry inspection"}}
    if tool_name == "tool_web_investigator" and "context" not in args:
        return {"context": args}
    return args


def _get_object_crop(state):
    full = state.get("full_image")
    bbox = state.get("detected_bbox")
    if full is not None and bbox and len(bbox) == 4:
        x, y, bw, bh = [int(v) for v in bbox]
        h_img, w_img = full.shape[:2]
        x = max(0, x); y = max(0, y)
        bw = min(bw, w_img - x); bh = min(bh, h_img - y)
        if bw > 5 and bh > 5:
            return full[y:y+bh, x:x+bw].copy()
    return full


def _generate_heatmap_async(state: dict):
    """Scanner léger pour heatmap GIF uniquement — ne modifie pas la décision."""
    if not SCANNER_AVAILABLE:
        return
    full = state.get("full_image")
    if full is None:
        return
    try:
        h, w = full.shape[:2]
        scale = 1.0
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            small = cv2.resize(full, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            small = full
        result = _scanner_fn(image_input=small, sensitivity="low")
        hm = result.get("heatmap")
        if hm is not None and isinstance(hm, np.ndarray):
            if scale != 1.0:
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            state["heatmap"] = hm
            print(f"  [Scanner] Heatmap OK for GIF: {hm.shape}")
    except Exception as e:
        print(f"  [Scanner] Heatmap skipped: {e}")


def _resolve(tool_name, args, state):
    args.pop("zone_index", None)
    resolved = {k: v for k, v in args.items()
                if k not in ("image_input", "patch", "patch_vlm", "full_image")}
    info = TOOL_REGISTRY.get(tool_name, {})
    if not info.get("needs_image"):
        if tool_name == "tool_alert_commander":
            resolved.setdefault("full_image",   state.get("full_image"))
            resolved.setdefault("heatmap",       state.get("heatmap"))
            resolved.setdefault("zones",         state.get("zones", []))
            resolved.setdefault("detected_bbox", state.get("detected_bbox"))
        return resolved
    full = state.get("full_image")
    if tool_name == "tool_direct_vlm_inspector":
        resolved["image_input"] = full
    elif tool_name == "tool_object_localizer":
        resolved["image_input"] = full
        objs = state.get("vlm_objects", [])
        if objs:
            best = max(objs, key=lambda o: {"high":3,"medium":2,"low":1}.get(o.get("confidence","low"), 0))
            resolved.setdefault("obj_type",      best.get("type", "unknown"))
            resolved.setdefault("obj_color",     best.get("color", "unknown"))
            resolved.setdefault("obj_shape",     best.get("shape", "unknown"))
            resolved.setdefault("obj_size",      best.get("size", "small"))
            resolved.setdefault("location_text", best.get("location_description", ""))
        resolved.setdefault("n_objects",   state.get("vlm_n_objects", 1))
        resolved.setdefault("crop_offset", state.get("crop_offset", [0, 0]))
    else:
        crop = _get_object_crop(state)
        resolved["image_input"] = crop if crop is not None else full
    return resolved


def _execute(tool_name, args, state):
    if tool_name not in TOOL_REGISTRY:
        return {}, f"Unknown '{tool_name}'. Valid: {list(TOOL_REGISTRY.keys())}"
    args     = _fix_args(tool_name, args)
    resolved = _resolve(tool_name, args, state)
    if TOOL_REGISTRY[tool_name].get("needs_image") and "image_input" not in resolved:
        return {}, f"No image for {tool_name}"
    try:
        result = TOOL_REGISTRY[tool_name]["fn"](**resolved)
    except Exception as e:
        return {}, f"ERROR {tool_name}: {e}\n{traceback.format_exc()[:400]}"

    if tool_name == "tool_direct_vlm_inspector":
        state["vlm_verdict"]    = result.get("overall_verdict")
        state["vlm_confidence"] = result.get("verdict_confidence")
        state["vlm_objects"]    = result.get("objects", [])
        state["vlm_n_objects"]  = result.get("n_objects", 0)
        state["crop_offset"]    = result.get("crop_offset", [0, 0])
        objs = result.get("objects", [])
        if objs:
            best = max(objs, key=lambda o: {"high":3,"medium":2,"low":1}.get(o.get("confidence","low"), 0))
            state["object_detected"] = best.get("type", "unknown")
            state["hazard_level"]    = best.get("hazard_level", "unknown")
        _generate_heatmap_async(state)
    elif tool_name == "tool_object_localizer":
        if result.get("found") and result.get("bbox"):
            state["detected_bbox"]       = result["bbox"]
            state["all_bboxes"]          = result.get("all_bboxes", [result["bbox"]])
            state["localization_method"] = result.get("method", "unknown")
            print(f"  [Brain] Primary bbox: {result['bbox']} | All: {result.get('all_bboxes')}")
        else:
            print(f"  [Brain] Localization failed")
    elif tool_name == "tool_anomaly_hunter":
        state.update({
            "anomaly_score":  result.get("score"),
            "is_anomaly":     result.get("is_anomaly"),
            "threshold_used": result.get("threshold_used"),
            "clean_mean":     result.get("clean_mean"),
        })
    elif tool_name == "tool_spectral_eye":
        state.update({
            "dominant_material":  result.get("dominant_material"),
            "material_signature": result.get("material_signature", {}),
        })
    elif tool_name == "tool_memory_search":
        state["memory_reliable"]      = result.get("reliable_match", False)
        state["suggested_root_cause"] = result.get("suggested_root_cause")
    elif tool_name == "tool_web_investigator":
        state["web_findings"] = str(result.get("regulatory_context", ""))[:300]

    state.setdefault("tool_results", {})[tool_name] = result

    def _clean(v, k=""):
        if isinstance(v, np.ndarray): return f"<ndarray {v.shape}>"
        if k in ("patch","patch_vlm","full_image","heatmap","image_input"): return "<image_data>"
        if isinstance(v, list): return [_clean(i) for i in v]
        if isinstance(v, dict): return {kk: _clean(vv, kk) for kk, vv in v.items()
                                        if kk not in ("patch","patch_vlm","image_input")}
        return v

    safe = _clean(result)
    obs  = json.dumps(safe, indent=2, default=str)
    return result, obs[:1200] + ("..." if len(obs) > 1200 else "")


def run_inspection(image_input, station: str = "unknown", max_steps: int = 12) -> dict:
    start = time.time(); timestamp = datetime.now().isoformat()
    print(f"\n{'='*60}\n ELMAZRAA v12 — VLM-First + PatchCore\n Station: {station}\n{'='*60}\n")

    state = {
        "full_image": None, "heatmap": None, "zones": [],
        "vlm_verdict": None, "vlm_confidence": None, "vlm_objects": [], "vlm_n_objects": 0,
        "crop_offset": [0, 0],
        "detected_bbox": None, "all_bboxes": [], "localization_method": None,
        "anomaly_score": None, "is_anomaly": None, "threshold_used": None, "clean_mean": None,
        "dominant_material": None, "material_signature": {},
        "object_detected": None, "hazard_level": None,
        "memory_reliable": False, "suggested_root_cause": None,
        "web_findings": "", "station": station, "tool_results": {}
    }

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return {"decision": "ERROR", "error": str(image_input), "timestamp": timestamp}
        state["full_image"] = img
        print(f"Image: {img.shape} | {Path(str(image_input)).name}")
    else:
        state["full_image"] = image_input

    h_img, w_img = state["full_image"].shape[:2]
    system = SYSTEM_PROMPT.format(tool_descriptions=_format_descs())
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Station='{station}' | Image={h_img}x{w_img}px\nCall tool_direct_vlm_inspector first."}
    ]

    step = 0
    while step < max_steps:
        step += 1
        print(f"\n[Step {step}/{max_steps}]")
        try:
            llm_out = _call_llm(messages)
        except Exception as e:
            print(f"[LLM ERROR] {e}"); break

        print(llm_out)
        messages.append({"role": "assistant", "content": llm_out})
        action, args = _parse(llm_out)

        if not action:
            messages.append({"role": "user", "content": "THOUGHT:...\nACTION:...\nARGS:{...}"}); continue

        if action == "FINAL_DECISION":
            decision = args.get("decision", "HUMAN_REVIEW")
            print(f"\nFINAL: {decision}")
            report = {
                **args, "timestamp": timestamp, "station": station,
                "anomaly_score": state.get("anomaly_score"), "is_anomaly": state.get("is_anomaly"),
                "threshold_used": state.get("threshold_used"), "clean_mean": state.get("clean_mean"),
                "vlm_verdict": state.get("vlm_verdict"), "vlm_n_objects": state.get("vlm_n_objects", 0),
                "vlm_objects": state.get("vlm_objects", []),
                "detected_bbox": state.get("detected_bbox"), "all_bboxes": state.get("all_bboxes", []),
                "localization_method": state.get("localization_method"),
                "web_findings": state.get("web_findings", ""),
                "material_signature": state.get("material_signature", {}),
                "subsurface_hint": {}, "color_profile": {}, "n_suspects": 0,
                "inspection_steps": step, "duration_seconds": round(time.time() - start, 2)
            }
            alert = tool_alert_commander(
                decision=decision, report=report,
                full_image=state.get("full_image"), heatmap=state.get("heatmap"),
                zones=state.get("zones", []), detected_bbox=state.get("detected_bbox"),
            )
            memory = tool_memory_writer(report)
            report["alert"] = alert; report["memory"] = memory
            dur = round(time.time() - start, 2)
            print(f"\nDone {dur}s — {step} steps — {decision}")
            if alert.get("gif_path"): print(f"GIF: {alert['gif_path']}")
            if alert.get("report_path"): print(f"Report: {alert['report_path']}")
            return report

        if action in TOOL_REGISTRY:
            print(f"[{action}]")
            _, obs = _execute(action, args, state)
            print(obs)
            messages.append({"role": "user", "content": f"OBSERVATION from {action}:\n{obs}"})
        else:
            messages.append({"role": "user", "content": f"'{action}' unknown. Valid: {list(TOOL_REGISTRY.keys())} or FINAL_DECISION."})

    report = {
        "decision": "HUMAN_REVIEW", "object_detected": state.get("object_detected", "unknown"),
        "dominant_material": "unknown", "hazard_level": "unknown", "root_cause": "max_steps_reached",
        "station": station, "brain_reasoning": "Max steps reached.", "confidence": "low",
        "timestamp": timestamp, "anomaly_score": state.get("anomaly_score"),
        "threshold_used": state.get("threshold_used"), "n_suspects": 0,
        "inspection_steps": step, "duration_seconds": round(time.time() - start, 2),
        "detected_bbox": state.get("detected_bbox"), "all_bboxes": state.get("all_bboxes", []),
        "vlm_objects": state.get("vlm_objects", []),
    }
    tool_alert_commander(decision="HUMAN_REVIEW", report=report,
                         full_image=state.get("full_image"), heatmap=state.get("heatmap"),
                         zones=[], detected_bbox=state.get("detected_bbox"))
    tool_memory_writer(report)
    return report