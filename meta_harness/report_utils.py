"""Report parsing and validation helpers."""

from __future__ import annotations

import re
from typing import Any

GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
REASON_RE = re.compile(
    r"\[REASON\]\s*:\s*(.*?)(?=\n\s*###\s+ANOMALY_|\n\s*---\s*\n\s*##\s+SUMMARY|\Z)",
    re.DOTALL,
)
SUMMARY_RE = re.compile(r"##\s+SUMMARY\s*(.*?)(?=\n\s*---\s*\n\*\*END OF REPORT\*\*|\Z)", re.DOTALL)
CONCLUSION_RE = re.compile(r"\[Conclusion\]\s*:?\*?\*?\s*:?\s*(FORGED|AUTHENTIC|REAL)", re.IGNORECASE)
RISK_RE = re.compile(r"\[RISK_SCORE\]\s*:?\*?\*?\s*:?\s*(\d{1,3})", re.IGNORECASE)


def extract_conclusion(report: str) -> str:
    match = CONCLUSION_RE.search(report)
    if not match:
        return "UNKNOWN"
    value = match.group(1).upper()
    return "AUTHENTIC" if value == "REAL" else value


def extract_risk_score(report: str) -> int | None:
    match = RISK_RE.search(report)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def extract_reasons(report: str) -> list[str]:
    return [" ".join(match.group(1).split()) for match in REASON_RE.finditer(report)]


def extract_summary(report: str) -> str:
    match = SUMMARY_RE.search(report)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def extract_explanation_text(report: str) -> str:
    parts = extract_reasons(report)
    summary = extract_summary(report)
    if summary:
        parts.append(summary)
    return "\n\n".join(part for part in parts if part)


def extract_groundings(report: str) -> list[tuple[int, int, int, int]]:
    return [tuple(int(v) for v in match.groups()) for match in GROUNDING_RE.finditer(report)]


def validate_report_schema(report: str) -> dict[str, Any]:
    conclusion = extract_conclusion(report)
    risk_score = extract_risk_score(report)
    groundings = extract_groundings(report)
    reasons = extract_reasons(report)
    errors: list[str] = []
    if "# FORGERY ANALYSIS" not in report:
        errors.append("missing_header")
    if conclusion == "UNKNOWN":
        errors.append("missing_conclusion")
    if risk_score is None:
        errors.append("missing_risk_score")
    if "## DETAILED ANOMALY ANALYSIS" not in report:
        errors.append("missing_anomaly_section")
    if "## SUMMARY" not in report:
        errors.append("missing_summary")
    if "**END OF REPORT**" not in report:
        errors.append("missing_end_marker")
    if conclusion == "FORGED" and not reasons:
        errors.append("forged_without_reasons")
    if conclusion == "FORGED" and len(groundings) < len(reasons):
        errors.append("fewer_groundings_than_reasons")
    return {
        "valid": not errors,
        "errors": errors,
        "conclusion": conclusion,
        "risk_score": risk_score,
        "num_reasons": len(reasons),
        "num_groundings": len(groundings),
    }


def build_authentic_report(summary: str) -> str:
    return f"""# FORGERY ANALYSIS  REPORT

**Report ID:** FAR-xxxx-xx-xx
**Date of Examination:** xxxx-xx-xx
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 0

---

## DETAILED ANOMALY ANALYSIS

No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.

---

## SUMMARY
{summary}

---
**END OF REPORT**"""


def build_forged_report(anomalies: list[dict[str, Any]], risk_score: int, summary: str) -> str:
    sections = []
    for idx, anomaly in enumerate(anomalies, 1):
        box = anomaly.get("box", [0, 0, 0, 0])
        title = anomaly.get("title", "Predicted Forgery Region")
        kind = anomaly.get("kind", "Visual Clumsy")
        reason = anomaly.get("reason", "The highlighted region is indicated by the provided forgery mask as manipulated and should be inspected for local visual or semantic inconsistencies.")
        sections.append(
            f"### ANOMALY_{idx:03d}: {kind} ({title})\n"
            f"[GROUNDING]:[{box[0]}, {box[1]}, {box[2]}, {box[3]}]\n"
            f"[REASON]: {reason}"
        )
    anomaly_text = "\n\n".join(sections)
    return f"""# FORGERY ANALYSIS  REPORT

**Report ID:** FAR-xxxx-xx-xx
**Date of Examination:** xxxx-xx-xx
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** FORGED
    **[RISK_SCORE]:** {max(0, min(100, int(risk_score)))}

---

## DETAILED ANOMALY ANALYSIS

The following sections detail the specific tampered regions identified during the examination.

{anomaly_text}

---

## SUMMARY
{summary}

---
**END OF REPORT**"""
