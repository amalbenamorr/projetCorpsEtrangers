"""
dashboard_server.py — ELMAZRAA Real-Time Dashboard Server
==========================================================
Lance le dashboard + WebSocket real-time + API pour envoyer images test.

USAGE:
    pip install flask flask-socketio python-dotenv
    python dashboard_server.py

PORTS:
    http://localhost:5500  ← Dashboard principal
    http://localhost:5500/test  ← Page envoi images test
"""

import json
import os
import base64
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "elmazraa-secret-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

OUTPUTS_DIR = Path("outputs")
LOGS_DIR    = Path("logs")
OUTPUTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
ALERT_LOG = LOGS_DIR / "alerts.log"

# ── READ ALERTS LOG ───────────────────────────────────────────────────────────
def read_alerts(limit=100):
    if not ALERT_LOG.exists():
        return []
    entries = []
    with open(ALERT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except:
                pass
    return list(reversed(entries))[-limit:]


def get_stats():
    alerts = read_alerts(500)
    total  = len(alerts)
    reject = sum(1 for a in alerts if a.get("decision") == "REJECT")
    review = sum(1 for a in alerts if a.get("decision") == "HUMAN_REVIEW")
    accept = sum(1 for a in alerts if a.get("decision") == "ACCEPT")
    rate   = round((reject + review) / total * 100, 1) if total else 0
    return {
        "total": total, "reject": reject,
        "review": review, "accept": accept, "rate": rate
    }


# ── WATCH ALERTS LOG FOR CHANGES ──────────────────────────────────────────────
_last_size = 0

def watch_log():
    global _last_size
    while True:
        time.sleep(2)
        try:
            if not ALERT_LOG.exists():
                continue
            size = ALERT_LOG.stat().st_size
            if size != _last_size:
                _last_size = size
                alerts = read_alerts(50)
                stats  = get_stats()
                latest = alerts[0] if alerts else {}
                socketio.emit("update", {
                    "alerts": alerts[:30],
                    "stats":  stats,
                    "latest": latest
                })
        except Exception as e:
            print(f"[watcher] {e}")


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return send_from_directory("static", "elmazraa_dashboard.html")


@app.route("/test")
def test_page():
    return send_from_directory("static", "elmazraa_test.html")


@app.route("/api/alerts")
def api_alerts():
    limit = int(request.args.get("limit", 50))
    return jsonify(read_alerts(limit))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory("outputs", filename)


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """
    Endpoint pour envoyer une image et lancer l'inspection.
    Body: multipart/form-data avec 'image' + optionnel 'station'
    """
    import sys, cv2, numpy as np
    sys.path.insert(0, str(Path(__file__).parent))

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file    = request.files["image"]
    station = request.form.get("station", "Station-Test")

    # Save temp image
    tmp_path = OUTPUTS_DIR / f"test_input_{datetime.now().strftime('%H%M%S')}.jpg"
    file.save(str(tmp_path))

    def run_inspection():
        try:
            from agents.master_agent import run_inspection
            result = run_inspection(str(tmp_path), station=station)
            # Emit to all connected clients
            socketio.emit("inspection_done", {
                "result": _safe(result),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"[INSPECTION ERROR] {err}")
            socketio.emit("inspection_error", {"error": str(e), "detail": err})
        finally:
            try:
                tmp_path.unlink()
            except:
                pass

    thread = threading.Thread(target=run_inspection)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "running", "station": station})


def _safe(obj):
    """Remove numpy arrays and non-serializable objects."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()
                if k not in ("full_image", "heatmap", "patch", "image_input")}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return f"<array {obj.shape}>"
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ── WHATSAPP TEST ENDPOINT ────────────────────────────────────────────────────
@app.route("/api/whatsapp/test", methods=["POST"])
def wa_test():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from tools.tool_whatsapp_notifier import tool_whatsapp_notifier
        result = tool_whatsapp_notifier(
            decision="REJECT",
            report={
                "id": "TEST001",
                "object_detected": "metal_fragment",
                "dominant_material": "metal",
                "hazard_level": "critical",
                "station": "Station-Test",
                "anomaly_score": 0.847,
                "brain_reasoning": "Test depuis dashboard ELMAZRAA."
            },
            force=True
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── SOCKETIO ──────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    alerts = read_alerts(30)
    stats  = get_stats()
    emit("update", {"alerts": alerts, "stats": stats, "latest": alerts[0] if alerts else {}})


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ELMAZRAA Dashboard Server")
    print("  Dashboard  → http://localhost:5500")
    print("  Test Page  → http://localhost:5500/test")
    print("=" * 55)

    watcher = threading.Thread(target=watch_log, daemon=True)
    watcher.start()

    # socketio.run(app, host="0.0.0.0", port=5500, debug=False, allow_unsafe_werkzeug=True)
    port = int(os.environ.get("PORT", 5500))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)