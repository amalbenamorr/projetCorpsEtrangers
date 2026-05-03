"""
tool_memory_writer.py
─────────────────────
"Write this in your notebook."

After every case, the agent writes what it learned to ChromaDB.
This is how the system becomes smarter over time — auto-learning RAG.

The brain calls this at the END of every inspection.
It never skips writing — even ACCEPT cases are valuable.
"""

import json
import uuid
import os
from datetime import datetime
from pathlib import Path

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

MEMORY_DIR      = "memory"
COLLECTION_NAME = "foreign_object_cases"
LOG_FILE        = "logs/memory_writes.log"


def _get_client():
    Path(MEMORY_DIR).mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=MEMORY_DIR)
    return client


def _get_collection(client):
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


def _build_document_text(case: dict) -> str:
    """Build rich text representation for embedding."""
    parts = [
        f"Foreign object detection case.",
        f"Object: {case.get('object_detected', 'unknown')}.",
        f"Material: {case.get('dominant_material', 'unknown')}.",
        f"Shape: {case.get('shape', 'unknown')}.",
        f"Color: {case.get('color', 'unknown')}.",
        f"Hazard level: {case.get('hazard_level', 'unknown')}.",
        f"Decision: {case.get('decision', 'unknown')}.",
        f"Root cause: {case.get('root_cause', 'unknown')}.",
        f"Station: {case.get('station', 'unknown')}.",
    ]
    if case.get("description"):
        parts.append(f"Visual description: {case['description'][:300]}.")
    if case.get("brain_reasoning"):
        parts.append(f"Agent reasoning: {case['brain_reasoning'][:200]}.")
    return " ".join(parts)


def tool_memory_writer(case: dict) -> dict:
    """
    Writes a completed inspection case to ChromaDB memory.

    Args:
        case : dict with all collected data from this inspection:
            - object_detected      : str
            - dominant_material    : str (from spectral eye)
            - shape                : str (from forensic brain)
            - color                : str
            - hazard_level         : str
            - decision             : str ("ACCEPT" | "REJECT" | "HUMAN_REVIEW")
            - root_cause           : str (agent's hypothesis)
            - station              : str (production line station, if known)
            - description          : str (from forensic brain)
            - anomaly_score        : float (from anomaly hunter)
            - brain_reasoning      : str (master agent's final reasoning)
            - web_findings         : str (from web investigator, if called)

    Returns:
        success    : bool
        case_id    : str (UUID of the stored case)
        verdict    : str
    """
    if not CHROMA_AVAILABLE:
        return {
            "success":  False,
            "case_id":  None,
            "error":    "chromadb_not_installed",
            "verdict":  "memory_write_failed — chromadb not installed"
        }

    try:
        client     = _get_client()
        collection = _get_collection(client)

        case_id   = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Sanitize — all values must be strings for ChromaDB metadata
        metadata = {
            "case_id":          case_id,
            "timestamp":        timestamp,
            "object_detected":  str(case.get("object_detected", "unknown")),
            "dominant_material": str(case.get("dominant_material", "unknown")),
            "shape":            str(case.get("shape", "unknown")),
            "color":            str(case.get("color", "unknown")),
            "hazard_level":     str(case.get("hazard_level", "unknown")),
            "decision":         str(case.get("decision", "unknown")),
            "root_cause":       str(case.get("root_cause", "unknown")),
            "station":          str(case.get("station", "unknown")),
            "anomaly_score":    str(case.get("anomaly_score", "0")),
            "web_findings":     str(case.get("web_findings", ""))[:500],
            "brain_reasoning":  str(case.get("brain_reasoning", ""))[:500],
        }

        document = _build_document_text(case)

        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[case_id]
        )

        # Also write to log file
        Path("logs").mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_entry = {
                "case_id":   case_id,
                "timestamp": timestamp,
                "decision":  metadata["decision"],
                "object":    metadata["object_detected"],
                "hazard":    metadata["hazard_level"],
            }
            f.write(json.dumps(log_entry) + "\n")

        total = collection.count()

        return {
            "success":       True,
            "case_id":       case_id,
            "timestamp":     timestamp,
            "total_in_memory": total,
            "verdict":       f"case_written id={case_id} total_memory={total}"
        }

    except Exception as e:
        return {
            "success": False,
            "case_id": None,
            "error":   str(e),
            "verdict": f"memory_write_error: {str(e)}"
        }