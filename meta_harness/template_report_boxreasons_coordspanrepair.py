"""Template-first harness with coordinate-span parsing repair."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from meta_harness.harness import ForgeryReportHarness
from meta_harness.llm_clients import image_to_data_url
from meta_harness.overlay import connected_component_boxes, load_binary_mask, render_red_box_overlay
from meta_harness.report_utils import build_authentic_report, build_forged_report


class TemplateReportBoxreasonsCoordspanrepair(ForgeryReportHarness):
    ALLOWED_FORGERY_TYPES = ("visual_clumsy", "logical_fraud", "semantic_subtle")

    def __init__(self, model_client, config: dict[str, Any] | None = None):
        super().__init__(model_client, config)
        self._state = "{}"

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" -:;,.\n\t")
        text = re.sub(r"^(?:summary|conclusion|overall assessment)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:box|region|anomaly)\s*#?\s*\d+\s*[:\-)\]]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:visual observation|observation|logical analysis|analysis)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        if not text:
            return "The highlighted region shows localized signs of document manipulation."
        if len(text) > 420:
            text = text[:420].rsplit(" ", 1)[0].rstrip(" -:;,. ")
        return text or "The highlighted region shows localized signs of document manipulation."

    def _clean_summary(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" -:;,.\n\t")
        text = re.sub(r"^(?:summary|conclusion|overall assessment)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        if len(text) > 1600:
            text = text[:1600].rsplit(" ", 1)[0].rstrip(" -:;,. ")
        return text

    def _classify_forgery_type(self, reason: str) -> str:
        text = reason.lower()
        semantic_terms = (
            "semantic",
            "wording",
            "label",
            "title",
            "plural",
            "spelling",
            "phrase",
            "term",
            "subtle",
            "replacement",
            "substitution",
        )
        logical_terms = (
            "logical",
            "date",
            "number",
            "amount",
            "price",
            "field",
            "legal",
            "contradiction",
            "inconsistency",
            "non-existent",
            "organization",
            "scope",
            "classification",
            "zip code",
            "policy",
        )
        if any(term in text for term in semantic_terms):
            return "semantic_subtle"
        if any(term in text for term in logical_terms):
            return "logical_fraud"
        return "visual_clumsy"

    def _build_forged_summary(self, reasons: list[str], risk: int) -> str:
        prefix = (
            f"The examination of the document has identified {len(reasons)} distinct anomalies "
            f"that collectively raise a fraud risk score of {risk}."
        )
        if not reasons:
            return prefix
        condensed = []
        for reason in reasons[:6]:
            cleaned = self._clean_summary(reason)
            if cleaned:
                condensed.append(cleaned.rstrip("."))
        if not condensed:
            return prefix
        return f"{prefix} The detected forgeries include " + "; ".join(condensed) + "."

    def _build_authentic_summary(self, sample: dict[str, Any]) -> str:
        prefix = "The examination of the document has identified 0 anomalies, resulting in a fraud risk score of 0."
        image_url = image_to_data_url(sample["image_path"])
        prompt = (
            "Describe this document in 4-6 factual sentences. Mention the visible title or subject, the main content, "
            "and end by noting that the typography, layout, and alignment appear consistent and authentic. "
            "Return only the description paragraph."
        )
        response = self.call_model([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ])
        description = self._clean_summary(response)
        if not description:
            description = "The visible document layout, typography, and alignment appear consistent throughout, supporting authenticity."
        return f"{prefix} {description}"

    def _parse(self, text: str, n_boxes: int) -> tuple[list[str], str]:
        reasons = [""] * n_boxes
        summary = ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(?:[-*•]\s*)?(?:box|region|anomaly)?\s*#?\s*(\d+)\s*(?:[:\-)\]]|\s+[-–—]\s+|\s*[:=]\s*)\s*(.+)$", line, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= n_boxes:
                    reasons[idx - 1] = self._clean(m.group(2))
                continue
            m = re.match(r"^\[(\d+)\]\s*(.+)$", line)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= n_boxes:
                    reasons[idx - 1] = self._clean(m.group(2))
                continue
            if line.lower().startswith(("summary", "conclusion", "overall assessment")):
                summary = self._clean_summary(line.split(":", 1)[1] if ":" in line else line)
        reason_blocks = [
            self._clean(block)
            for block in re.findall(
                r"\[REASON\]\s*:\s*(.*?)(?=\n\s*###\s+ANOMALY_|\n\s*---\s*\n\s*##\s+SUMMARY|\n\s*##\s+SUMMARY|\Z)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        for idx, reason in enumerate(reason_blocks[:n_boxes]):
            if not reasons[idx]:
                reasons[idx] = reason
        if not summary:
            summary_match = re.search(r"##\s+SUMMARY\s*(.*?)(?=\n\s*---|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary = self._clean_summary(summary_match.group(1))
        fallback = self._clean(text)
        reasons = [r or fallback for r in reasons]
        if not summary:
            summary = self._clean_summary(text)
        return reasons, summary

    def predict(self, sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        precomputed_boxes = sample.get("precomputed_boxes")
        precomputed_mask_empty = sample.get("precomputed_mask_empty")
        overlay_image_url = sample.get("precomputed_overlay_image_url")
        overlay_path = sample.get("precomputed_overlay_path")

        if precomputed_boxes is not None or precomputed_mask_empty is not None:
            boxes = list(precomputed_boxes or [])
        else:
            mask = sample.get("_loaded_mask")
            if mask is None and sample.get("mask_path"):
                mask = load_binary_mask(sample.get("mask_path"))

            boxes = connected_component_boxes(mask) if mask is not None else []

        if precomputed_mask_empty or not boxes:
            summary = self._build_authentic_summary(sample)
            return build_authentic_report(summary), {"boxes": [], "full_response": summary}

        if not overlay_image_url and overlay_path is not None:
            overlay_image_url = image_to_data_url(overlay_path)

        if not overlay_image_url:
            overlay_dir = Path(self.config.get("overlay_dir", "logs/overlays")) / "template_report_boxreasons_coordspanrepair"
            overlay_path = render_red_box_overlay(sample["image_path"], boxes, overlay_dir / f"{sample['sample_id']}.png")
            overlay_image_url = image_to_data_url(overlay_path)

        prompt = (
            "Inspect the document regions marked by the red boxes and explain the underlying anomaly in each region, not the red markup itself.\n"
            f"Use these coordinates only and keep the same order: {[b['box'] for b in boxes]}.\n"
            "Use a formal forensic tone that mirrors document-examination reports. For each box, write 1-2 short sentences: "
            "first state the local visual cue, then state the likely manipulation or forensic implication. Avoid saying only 'red box', "
            "'highlighted area', or 'text is obscured'. Describe the actual issue such as pixelation, pasted text, inconsistent font weight, "
            "misalignment, erased content, replacement, unnatural spacing, contrast mismatch, or semantic inconsistency.\n"
            "The anomaly type in the final report must be one of: visual_clumsy, logical_fraud, semantic_subtle.\n"
            "Stay conservative: only mention content you can visually support from the boxed region.\n"
            "Format exactly as:\n"
            "1: <forensic reason for box 1>\n"
            "2: <forensic reason for box 2>\n"
            "Summary: <brief summary of all detected forgeries>\n"
            "Example style:\n"
            "1: The boxed text shows jagged masking and a tonal break from the surrounding print, suggesting a clumsy digital redaction over the original entry.\n"
            "2: The characters are slightly misaligned and differ in weight from adjacent text, consistent with a local text replacement rather than native document rendering.\n"
            "Summary: The document contains localized text and texture inconsistencies consistent with targeted tampering across the highlighted regions."
        )
        response = self.call_model([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": overlay_image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ])
        
        reasons, summary = self._parse(response, len(boxes))
        anomalies = [
            {
                "box": b["box"],
                "title": f"Region {i}",
                "kind": self._classify_forgery_type(reasons[i - 1]),
                "reason": reasons[i - 1],
            }
            for i, b in enumerate(boxes, 1)
        ]
        risk = min(100, len(boxes) * 10)
        report = build_forged_report(anomalies, risk, self._build_forged_summary(reasons, risk))
        return report, {
            "boxes": boxes,
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
            "full_response": response,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        return None

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state
