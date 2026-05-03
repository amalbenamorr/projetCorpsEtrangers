"""
tool_web_investigator.py
────────────────────────
"I've never seen this before. Let me check the internet."

Uses Tavily to search for:
- Regulatory context (food safety standards)
- Similar contamination cases in the industry
- Material identification support
- Root cause patterns globally

The brain calls this ONLY when:
- Memory search returned nothing
- Hazard level is unknown/critical
- The object is completely unidentified
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def _build_search_query(context: dict) -> str:
    """Build an intelligent, specific Tavily query from detection context."""
    base = "poultry processing foreign object contamination"

    parts = []
    if context.get("object_detected") and context["object_detected"] != "unknown":
        parts.append(context["object_detected"])
    if context.get("dominant_material") and context["dominant_material"] not in ("unknown", "organic"):
        parts.append(context["dominant_material"])
    if context.get("shape") and context["shape"] != "unknown":
        parts.append(context["shape"])
    if context.get("color") and context["color"] not in ("unknown", "pink", "red"):
        parts.append(context["color"])

    if parts:
        return f"{base} {' '.join(parts)} food safety hazard"
    return f"{base} food safety regulatory"


def _extract_insights(results: list) -> dict:
    """
    Parse Tavily search results into structured insights.
    No hardcoded rules — just aggregate what was found.
    """
    regulatory_refs  = []
    root_cause_hints = []
    severity_hints   = []
    action_hints     = []
    sources          = []

    keywords_regulatory  = ["regulation", "standard", "HACCP", "FDA", "USDA", "EU", "directive", "compliance"]
    keywords_root_cause  = ["caused by", "due to", "worn", "broken", "failure", "source", "origin"]
    keywords_severity    = ["critical", "recall", "reject", "hazard", "risk", "contamination", "alert"]
    keywords_action      = ["recall", "shutdown", "inspection", "quarantine", "reject", "stop"]

    for r in results:
        content = r.get("content", "") or ""
        url     = r.get("url", "")
        title   = r.get("title", "")

        sources.append({"title": title[:80], "url": url[:120]})

        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sent in sentences:
            sent_l = sent.lower()
            if any(k in sent_l for k in keywords_regulatory):
                regulatory_refs.append(sent[:200])
            if any(k in sent_l for k in keywords_root_cause):
                root_cause_hints.append(sent[:200])
            if any(k in sent_l for k in keywords_severity):
                severity_hints.append(sent[:200])
            if any(k in sent_l for k in keywords_action):
                action_hints.append(sent[:200])

    return {
        "regulatory_context": regulatory_refs[:3],
        "root_cause_hints":   root_cause_hints[:3],
        "severity_hints":     severity_hints[:3],
        "action_hints":       action_hints[:3],
        "sources":            sources[:5]
    }


def tool_web_investigator(context: dict, custom_query: str = None) -> dict:
    """
    Answers: "What does the world know about this?"

    Args:
        context      : dict with keys like object_detected, dominant_material,
                       description, hazard_level from other tools
        custom_query : optional override query string

    Returns:
        findings           : dict of categorized insights
        regulatory_context : list of relevant regulatory sentences
        root_cause_hints   : list of possible root causes
        severity_context   : list of severity-related findings
        recommended_action : str (aggregate from web)
        sources            : list of source URLs
        query_used         : str
        verdict            : str for brain
    """
    if not TAVILY_AVAILABLE:
        return {
            "findings":            {},
            "regulatory_context":  [],
            "root_cause_hints":    [],
            "severity_context":    [],
            "recommended_action":  "web_search_unavailable",
            "sources":             [],
            "query_used":          "",
            "error":               "tavily_not_installed",
            "verdict":             "web_investigator_unavailable"
        }

    if not TAVILY_API_KEY:
        return {
            "findings":            {},
            "regulatory_context":  [],
            "root_cause_hints":    [],
            "severity_context":    [],
            "recommended_action":  "no_api_key",
            "sources":             [],
            "query_used":          "",
            "error":               "TAVILY_API_KEY not set",
            "verdict":             "web_search_disabled — no API key"
        }

    query = custom_query or _build_search_query(context)

    try:
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "fda.gov", "usda.gov", "efsa.europa.eu",
                "foodsafetynews.com", "food-safety.com",
                "poultryworld.net", "thepoultrysite.com"
            ]
        )

        raw_results = results.get("results", [])
        insights    = _extract_insights(raw_results)

        # Aggregate recommended action from hints
        all_actions = insights["action_hints"] + insights["severity_hints"]
        recommended_action = all_actions[0] if all_actions else "consult_standard_procedure"

        verdict = (
            f"web_search_complete query='{query[:50]}' "
            f"sources={len(raw_results)} "
            f"regulatory_refs={len(insights['regulatory_context'])}"
        )

        return {
            "findings":            insights,
            "regulatory_context":  insights["regulatory_context"],
            "root_cause_hints":    insights["root_cause_hints"],
            "severity_context":    insights["severity_hints"],
            "recommended_action":  recommended_action,
            "sources":             insights["sources"],
            "query_used":          query,
            "verdict":             verdict
        }

    except Exception as e:
        return {
            "findings":            {},
            "regulatory_context":  [],
            "root_cause_hints":    [],
            "severity_context":    [],
            "recommended_action":  "error",
            "sources":             [],
            "query_used":          query,
            "error":               str(e),
            "verdict":             f"web_search_failed: {str(e)}"
        }