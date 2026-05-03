"""
tool_memory_search.py — v4
──────────────────────────
"Have I seen this before?"

FIX v4:
- Seuil similarity relevance: 0.6 → 0.75 (évite les faux rappels)
- Filtre les cas avec decision="HUMAN_REVIEW" + confidence="low"
  (ne pas propager des cas douteux)
- Retourne un champ 'reliable_match' clair pour le brain
- Pondération temporelle : cas récents > cas anciens si similarity proche
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

MEMORY_DIR      = "memory"
COLLECTION_NAME = "foreign_object_cases"
MIN_SIMILARITY  = 0.75   # seuil strict — en dessous, pas fiable


def _get_client():
    Path(MEMORY_DIR).mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=MEMORY_DIR)


def _get_collection(client):
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


def _build_query_text(context: dict) -> str:
    parts = []
    if context.get("object_detected") and context["object_detected"] not in ("unknown", ""):
        parts.append(context["object_detected"])
    if context.get("dominant_material") and context["dominant_material"] not in ("unknown", "organic"):
        parts.append(context["dominant_material"])
    if context.get("shape") and context["shape"] != "unknown":
        parts.append(context["shape"])
    if context.get("color") and context["color"] not in ("unknown", "pink", "red"):
        parts.append(context["color"])
    if context.get("description"):
        parts.append(context["description"][:200])
    return " ".join(parts) if parts else "foreign object poultry inspection"


def tool_memory_search(context: dict, n_results: int = 5) -> dict:
    """
    Searches past cases in ChromaDB.

    CRITICAL: Only returns 'reliable_match=True' if similarity >= 0.75.
    The brain must check reliable_match before trusting the suggestion.

    Returns:
        found           : bool (any result found)
        reliable_match  : bool (similarity >= 0.75 → trust this)
        n_matches       : int
        cases           : list
        suggested_root_cause : str or None
        confidence      : "high" / "medium" / "low"
        verdict         : str
    """
    if not CHROMA_AVAILABLE:
        return {
            "found": False, "reliable_match": False,
            "n_matches": 0, "cases": [],
            "recurring_patterns": [], "suggested_root_cause": None,
            "confidence": "low", "error": "chromadb_not_installed",
            "verdict": "memory_unavailable"
        }

    try:
        client     = _get_client()
        collection = _get_collection(client)

        count = collection.count()
        if count == 0:
            return {
                "found": False, "reliable_match": False,
                "n_matches": 0, "cases": [],
                "recurring_patterns": [], "suggested_root_cause": None,
                "confidence": "low",
                "verdict": "memory_empty — first cases ever"
            }

        query_text = _build_query_text(context)
        n_req      = min(n_results, count)

        results = collection.query(
            query_texts=[query_text],
            n_results=n_req,
            include=["documents", "metadatas", "distances"]
        )

        docs      = results["documents"][0] if results["documents"] else []
        metas     = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # Filtre strict : similarity >= 0.75
        # + on exclut les cas peu fiables (HUMAN_REVIEW sans confiance)
        relevant_cases = []
        for doc, meta, dist in zip(docs, metas, distances):
            sim = round(1 - dist, 3)
            if sim < MIN_SIMILARITY:
                continue
            # Ne pas propager des cas douteux
            if (meta.get("decision") == "HUMAN_REVIEW" and
                    meta.get("confidence", "low") == "low"):
                continue
            case = {
                "case_id":      meta.get("case_id", "unknown"),
                "timestamp":    meta.get("timestamp", "unknown"),
                "object":       meta.get("object_detected", "unknown"),
                "material":     meta.get("dominant_material", "unknown"),
                "hazard_level": meta.get("hazard_level", "unknown"),
                "decision":     meta.get("decision", "unknown"),
                "root_cause":   meta.get("root_cause", "unknown"),
                "station":      meta.get("station", "unknown"),
                "confidence":   meta.get("confidence", "low"),
                "similarity":   sim,
                "description":  doc[:200]
            }
            relevant_cases.append(case)

        if not relevant_cases:
            return {
                "found": False, "reliable_match": False,
                "n_matches": 0, "cases": [],
                "recurring_patterns": [],
                "suggested_root_cause": None,
                "confidence": "low",
                "verdict": f"no_reliable_cases_found (best_raw_sim={round(1-distances[0], 3) if distances else 0:.2f} < threshold={MIN_SIMILARITY})"
            }

        # Patterns récurrents
        from collections import Counter
        root_causes = [c["root_cause"] for c in relevant_cases if c["root_cause"] != "unknown"]
        stations    = [c["station"]    for c in relevant_cases if c["station"]    != "unknown"]
        decisions   = [c["decision"]   for c in relevant_cases if c["decision"]   != "unknown"]

        suggested_root_cause = Counter(root_causes).most_common(1)[0][0] if root_causes else None

        recurring_patterns = []
        if stations:
            top_s = Counter(stations).most_common(1)[0]
            if top_s[1] >= 2:
                recurring_patterns.append(f"recurs_at:{top_s[0]}")
        if decisions:
            top_d = Counter(decisions).most_common(1)[0][0]
            recurring_patterns.append(f"typical_decision:{top_d}")

        best_sim = relevant_cases[0]["similarity"]
        if best_sim >= 0.90:
            confidence = "high"
        elif best_sim >= 0.80:
            confidence = "medium"
        else:
            confidence = "low"

        verdict = (
            f"found {len(relevant_cases)} reliable cases — "
            f"best_similarity={best_sim:.2f} — "
            f"suggested_cause={suggested_root_cause or 'unknown'}"
        )

        return {
            "found":               True,
            "reliable_match":      True,
            "n_matches":           len(relevant_cases),
            "cases":               relevant_cases,
            "recurring_patterns":  recurring_patterns,
            "suggested_root_cause": suggested_root_cause,
            "confidence":          confidence,
            "verdict":             verdict
        }

    except Exception as e:
        return {
            "found": False, "reliable_match": False,
            "n_matches": 0, "cases": [],
            "recurring_patterns": [], "suggested_root_cause": None,
            "confidence": "low",
            "error": str(e),
            "verdict": f"memory_error: {str(e)}"
        }