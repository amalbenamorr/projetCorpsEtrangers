"""
tool_whatsapp_notifier.py — VERSION FINALE CORRIGEE
─────────────────────────────────────────────────────
PROBLEMES CORRIGES:
1. GREENAPI_TO = numéro DESTINATAIRE (superviseur), PAS ton propre numéro
   → Ton numéro connecté = 21626561742 (celui scanné sur Green API)
   → GREENAPI_TO doit être le numéro du superviseur qui reçoit les alertes
   → Ex: GREENAPI_TO=21699123456@c.us  (numéro du superviseur)

2. GIF envoyé AVANT le message texte via sendFileByUpload

CONFIG .env CORRECTE:
    GREENAPI_INSTANCE=7107576363
    GREENAPI_TOKEN=608213231ae4403f9c62974c3eaf8210391d8a6abc8f4829ae
    GREENAPI_API_URL=https://7107.api.greenapi.com
    GREENAPI_TO=216XXXXXXXXX@c.us   ← numéro superviseur (PAS le tien !)

FORMAT numéro: indicatif sans + puis numéro sans 0 + @c.us
    Tunisie 216 : 21699123456@c.us
    France  33  : 33612345678@c.us
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
GREENAPI_INSTANCE = os.getenv("GREENAPI_INSTANCE", "7107576363")
GREENAPI_TOKEN    = os.getenv("GREENAPI_TOKEN",    "")
GREENAPI_API_URL  = os.getenv("GREENAPI_API_URL",  "https://7107.api.greenapi.com")
GREENAPI_TO       = os.getenv("GREENAPI_TO",       "")   # ← numéro SUPERVISEUR


# ── BUILD MESSAGE ─────────────────────────────────────────────────────────────
def build_message(decision: str, report: dict) -> str:
    obj       = report.get("object_detected", "unknown")
    mat       = report.get("dominant_material", "unknown")
    hazard    = report.get("hazard_level", "unknown")
    station   = report.get("station", "unknown")
    score     = report.get("anomaly_score")
    score_s   = f"{score:.3f}" if isinstance(score, float) else str(score or "—")
    case_id   = str(report.get("id", datetime.now().strftime("%H%M%S")))[:8]
    reasoning = str(report.get("brain_reasoning", "") or "")[:120]
    now       = datetime.now().strftime("%H:%M:%S")

    OBJ_LABELS = {
        "metal_fragment": "Metal", "bone_shard": "Os", "feather": "Plume",
        "plastic": "Plastique", "wood": "Bois", "rubber": "Caoutchouc",
        "glass": "Verre", "stone": "Pierre", "unknown": "Inconnu", "none": "Aucun",
    }
    HAZ_LABELS = {
        "critical": "CRITIQUE", "moderate": "MODERE",
        "low": "FAIBLE", "none": "AUCUN", "unknown": "INCONNU",
    }

    obj_label = OBJ_LABELS.get(obj, obj)
    haz_label = HAZ_LABELS.get(hazard, hazard)

    if decision == "REJECT":
        header = "CRITICAL ALERT - Corps Etranger Detecte"
        action = "Action automatique: LIGNE ARRETEE"
    elif decision == "HUMAN_REVIEW":
        header = "VERIFICATION HUMAINE REQUISE"
        action = "Action: Verification operateur necessaire"
    else:
        return ""

    lines = [
        f"*{header}*",
        "─────────────────────",
        f"*Objet    :* {obj_label}",
        f"*Matiere  :* {mat}",
        f"*Danger   :* {haz_label}",
        f"*Station  :* {station}",
        f"*Score    :* {score_s}",
        f"*Heure    :* {now}",
        "─────────────────────",
        f"*{action}*",
        f"*Case ID  :* {case_id}",
    ]
    if reasoning:
        lines.append(f"*IA :* {reasoning}")
    lines += [
        "─────────────────────",
        "*EL MAZRAA Visual Forensics*",
        "_Powered by ELMAZRAA Agentic AI v12_",
    ]
    return "\n".join(lines)


# ── SEND TEXT ─────────────────────────────────────────────────────────────────
def _send_text(message: str, to: str) -> dict:
    url  = f"{GREENAPI_API_URL}/waInstance{GREENAPI_INSTANCE}/sendMessage/{GREENAPI_TOKEN}"
    body = json.dumps({"chatId": to, "message": message}).encode("utf-8")
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read().decode())
        return {"success": True, "method": "text",
                "idMessage": resp.get("idMessage", "?"), "response": resp}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── SEND GIF (fichier) ────────────────────────────────────────────────────────
def _send_gif(gif_path: str, to: str, caption: str = "") -> dict:
    """
    Envoie le GIF via sendFileByUpload (multipart form-data).
    Green API DEVELOPER permet l'envoi de fichiers.
    """
    p = Path(gif_path)
    if not p.exists():
        return {"success": False, "error": f"GIF introuvable: {gif_path}"}
    if p.stat().st_size == 0:
        return {"success": False, "error": "GIF vide (0 bytes)"}

    url = f"{GREENAPI_API_URL}/waInstance{GREENAPI_INSTANCE}/sendFileByUpload/{GREENAPI_TOKEN}"

    try:
        with open(p, "rb") as f:
            file_data = f.read()
    except Exception as e:
        return {"success": False, "error": f"Lecture GIF: {e}"}

    boundary = "ElMazraaGIF2026xyz"

    def _field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    parts = [_field("chatId", to)]
    if caption:
        parts.append(_field("caption", caption))

    # Fichier GIF
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
            f"Content-Type: image/gif\r\n\r\n"
        ).encode("utf-8")
        + file_data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            resp = json.loads(r.read().decode())
        return {"success": True, "method": "gif-upload",
                "idMessage": resp.get("idMessage", "?"), "response": resp}
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        return {"success": False, "error": f"HTTP {e.code}: {err[:400]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── LOG ───────────────────────────────────────────────────────────────────────
def _log(decision: str, report: dict, result: dict):
    log_path = Path("logs") / "whatsapp_notifications.log"
    log_path.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "decision":  decision,
        "to":        GREENAPI_TO,
        "station":   report.get("station"),
        "object":    report.get("object_detected"),
        "success":   result.get("success"),
        "error":     result.get("error", ""),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── MAIN TOOL ─────────────────────────────────────────────────────────────────
def tool_whatsapp_notifier(
    decision: str,
    report:   dict,
    gif_path: str  = None,
    force:    bool = False,
) -> dict:
    """
    Envoie notification WhatsApp + GIF via Green API.

    IMPORTANT: GREENAPI_TO dans .env = numéro du SUPERVISEUR (pas ton propre numéro).

    Args:
        decision : REJECT | HUMAN_REVIEW | ACCEPT
        report   : rapport complet du master_agent
        gif_path : chemin vers outputs/xxx.gif (optionnel)
        force    : si True, envoie même pour ACCEPT (tests)
    """
    # ACCEPT → pas de notification sauf si forcé
    if decision == "ACCEPT" and not force:
        return {"sent": False, "success": True, "reason": "ACCEPT — pas de notification"}

    message = build_message(decision, report)
    if not message:
        return {"sent": False, "success": True, "reason": "no message built"}

    # ── Vérification config ───────────────────────────────────────────────────
    if not GREENAPI_TOKEN or not GREENAPI_TO:
        missing = []
        if not GREENAPI_TOKEN: missing.append("GREENAPI_TOKEN")
        if not GREENAPI_TO:    missing.append("GREENAPI_TO")
        print(f"\n[WhatsApp] Config manquante dans .env : {', '.join(missing)}")
        print("\n" + "=" * 55)
        print("SIMULATION:")
        print(message)
        print("=" * 55 + "\n")
        return {
            "sent":    False, "success": True,
            "method":  "simulation", "message": message,
            "note":    f"Ajoute {', '.join(missing)} dans .env",
        }

    # ── Vérification : ne pas s'envoyer à soi-même ───────────────────────────
    # Ton numéro connecté = 21626561742
    # Si GREENAPI_TO = 21626561742@c.us → erreur de config
    if "21626561742" in GREENAPI_TO:
        print("\n[WhatsApp] ATTENTION: GREENAPI_TO = ton propre numéro!")
        print("  Change GREENAPI_TO dans .env pour le numéro du superviseur.")
        print("  Ex: GREENAPI_TO=21699XXXXXX@c.us")
        print("  (Pour les tests tu peux laisser ton numéro, mais en prod change-le)\n")

    print(f"\n[WhatsApp] Envoi {decision} → {GREENAPI_TO}")

    gif_result  = None
    text_result = None

    # ── 1. Envoyer le GIF en premier ─────────────────────────────────────────
    if gif_path:
        print(f"  GIF : {gif_path}")
        size_kb = Path(gif_path).stat().st_size // 1024 if Path(gif_path).exists() else 0
        print(f"  Taille GIF : {size_kb} KB")
        caption = (
            f"ELMAZRAA | {decision} | "
            f"{report.get('object_detected','?')} @ {report.get('station','?')}"
        )
        gif_result = _send_gif(gif_path, GREENAPI_TO, caption=caption)
        if gif_result.get("success"):
            print(f"  GIF envoyé OK — id: {gif_result.get('idMessage','?')}")
        else:
            print(f"  GIF echoue: {gif_result.get('error')}")
    else:
        print("  Pas de GIF fourni")

    # ── 2. Message texte ──────────────────────────────────────────────────────
    print("  Envoi message texte...")
    text_result = _send_text(message, GREENAPI_TO)
    if text_result.get("success"):
        print(f"  Texte envoyé OK — id: {text_result.get('idMessage','?')}")
    else:
        print(f"  Texte echoue: {text_result.get('error')}")

    final = {
        "sent":        True,
        "method":      "green-api",
        "success":     text_result.get("success", False),
        "to":          GREENAPI_TO,
        "message":     message,
        "text_result": text_result,
        "gif_result":  gif_result,
        "gif_sent":    gif_result.get("success", False) if gif_result else False,
        "error":       text_result.get("error", "") if not text_result.get("success") else "",
    }

    _log(decision, report, final)
    return final


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("TEST — ELMAZRAA WhatsApp Notifier")
    print("=" * 55)
    print(f"Instance  : {GREENAPI_INSTANCE}")
    print(f"API URL   : {GREENAPI_API_URL}")
    print(f"Token     : {'SET ✓' if GREENAPI_TOKEN else 'NON SET ✗'}")
    print(f"Envoi à   : {GREENAPI_TO or 'NON SET ✗'}")

    if "21626561742" in GREENAPI_TO:
        print("\n  ATTENTION: tu t'envoies à toi-même (test OK mais en prod change GREENAPI_TO)")

    print()

    # Cherche un GIF de test dans outputs/
    test_gif = None
    outputs  = Path("outputs")
    if outputs.exists():
        gifs = list(outputs.glob("*.gif"))
        if gifs:
            test_gif = str(gifs[0])
            print(f"GIF trouvé pour test : {test_gif}")
        else:
            print("Pas de GIF dans outputs/ — envoi texte seulement")

    test_report = {
        "id":                "TEST001",
        "object_detected":   "metal_fragment",
        "dominant_material": "metal",
        "hazard_level":      "critical",
        "station":           "Station-3",
        "anomaly_score":     0.847,
        "brain_reasoning":   "DirectVLM=REJECT conf=high. Metal fragment on meat surface.",
    }

    result = tool_whatsapp_notifier(
        decision="REJECT",
        report=test_report,
        gif_path=test_gif,
        force=True,
    )

    print("\nRESULTAT:")
    for k, v in result.items():
        if k not in ("message", "text_result", "gif_result"):
            print(f"  {k}: {v}")
    if result.get("text_result"):
        print(f"\n  text_result: {json.dumps(result['text_result'], indent=4)}")
    if result.get("gif_result"):
        print(f"\n  gif_result:  {json.dumps(result['gif_result'], indent=4)}")