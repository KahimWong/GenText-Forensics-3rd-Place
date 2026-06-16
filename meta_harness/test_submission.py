"""Generate submission reports for the RealText-V2 test set."""

from __future__ import annotations

import os

os.environ.setdefault("LINKAPI_API_KEY", os.environ.get("LINKAPI_API_KEY", ""))
import argparse
import gzip
import importlib
import inspect
import json
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from meta_harness.harness import ForgeryReportHarness
from meta_harness.llm_clients import (
    DEFAULT_MAX_DATA_URI_CHARS,
    ModelStudioChatClient,
    image_to_data_url,
    make_stub_client,
)
from meta_harness.overlay import connected_component_boxes, load_binary_mask
from meta_harness.report_utils import build_authentic_report, build_forged_report

ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_IMAGE_LIST = Path("data/test_image_path_list.pkl")
DEFAULT_MASK_ROOT = Path("data/infer_path_eval")
DEFAULT_OUTPUT = ROOT / "logs" / "test_submission" / "prediction.jsonl"
DEFAULT_LOG = ROOT / "logs" / "test_submission" / "trace.jsonl"
DEFAULT_PRECOMPUTED_ARTIFACTS = Path("data/infer_path_eval/precomputed_artifacts/artifacts")
AUTHENTIC_SUMMARY = "No mask-indicated forged regions are present; the document is assessed as authentic with a risk score of zero."
FALLBACK_REASON_TEMPLATE = (
    "The marked region shows localized inconsistencies relative to the surrounding document content, "
    "which is consistent with possible manipulation and warrants forensic review."
)
FALLBACK_SUMMARY_TEMPLATE = (
    "The document contains one or more marked regions with localized inconsistencies consistent with possible tampering. "
    "This fallback report was produced because automated generation was unavailable for this sample."
)
WORKER_COUNT = 64


def _is_data_inspection_failed_error(exc: Exception) -> bool:
    message = str(exc)
    return "data_inspection_failed" in message or "DataInspectionFailed" in message


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    message = str(exc)
    return (
        name == "APITimeoutError"
        or "Request timed out" in message
        or "timed out" in message.lower()
    )


def _build_fallback_forged_report(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    boxes = list(sample.get("precomputed_boxes") or [])
    if not boxes:
        mask = sample.get("_loaded_mask")
        if mask is None and sample.get("mask_path"):
            mask = load_binary_mask(sample["mask_path"])
        if mask is not None:
            boxes = connected_component_boxes(mask)
    if not boxes:
        boxes = [{"box": [0, 0, 0, 0], "area": 0}]
    anomalies = [
        {
            "box": box.get("box", [0, 0, 0, 0]),
            "title": f"Region {idx}",
            "kind": "visual_clumsy",
            "reason": FALLBACK_REASON_TEMPLATE,
        }
        for idx, box in enumerate(boxes, 1)
    ]
    risk_score = min(100, max(10, len(anomalies) * 10))
    return build_forged_report(anomalies, risk_score, FALLBACK_SUMMARY_TEMPLATE), {
        "boxes": boxes,
        "mask_empty": False,
        "used_fallback": True,
        "fallback_reason": "data_inspection_failed",
    }


def _predict_with_fallback(
    harness: ForgeryReportHarness, sample: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    try:
        return harness.predict(sample)
    except Exception as exc:
        if not (_is_data_inspection_failed_error(exc) or _is_timeout_error(exc)):
            raise
        fallback_reason = (
            "timeout" if _is_timeout_error(exc) else "data_inspection_failed"
        )
        if sample.get("precomputed_mask_empty"):
            return build_authentic_report(AUTHENTIC_SUMMARY), {
                "mask_empty": True,
                "used_fallback": True,
                "fallback_reason": fallback_reason,
            }
        report, metadata = _build_fallback_forged_report(sample)
        return report, {**metadata, "fallback_reason": fallback_reason}


class JSONLLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.start_time = time.time()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def log(self, type: str, **data: Any) -> None:
        if not self.path:
            return
        entry = {"type": type, "t": round(time.time() - self.start_time, 2), **data}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_trace_samples(
    log_path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    samples_map: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {}
    if not log_path:
        return samples_map, meta
    path = Path(log_path)
    if not path.is_file():
        return samples_map, meta
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type")
            if entry_type == "meta" and not meta:
                meta = entry
                continue
            if entry_type != "sample":
                continue
            image_name = entry.get("image_name")
            report = entry.get("report")
            if image_name and report:
                samples_map[image_name] = entry
    return samples_map, meta


def _read_precomputed_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty precomputed artifact file: {path}")


def _resolve_precomputed_overlay(
    precomputed: dict[str, Any],
) -> tuple[str | None, str | None]:
    overlay_path_raw = precomputed.get("overlay_path")
    overlay_path = str(overlay_path_raw) if overlay_path_raw else None
    overlay_image_url = precomputed.get("overlay_image_url")
    if isinstance(overlay_image_url, str) and overlay_image_url.startswith("data:"):
        if len(overlay_image_url) > DEFAULT_MAX_DATA_URI_CHARS:
            overlay_image_url = (
                image_to_data_url(overlay_path) if overlay_path else None
            )
    elif not overlay_image_url and overlay_path:
        overlay_image_url = image_to_data_url(overlay_path)
    return overlay_image_url, overlay_path


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_harness(
    path: str, model_client, config: dict[str, Any]
) -> ForgeryReportHarness:
    if "/" not in path and not path.endswith(".py"):
        return load_harness(f"agents/{path}.py", model_client, config)
    module_path = path.replace("/", ".").replace(".py", "")
    module = importlib.import_module(f"FIE_harness.{module_path}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ForgeryReportHarness) and obj is not ForgeryReportHarness:
            return obj(model_client=model_client, config=config)
    raise ValueError(f"No ForgeryReportHarness subclass found in {path}")


def _make_base_client(cfg: dict[str, Any], args: argparse.Namespace):
    if args.stub_response is not None:
        return make_stub_client(args.stub_response)
    base_cfg = cfg["models"]["base"]
    return ModelStudioChatClient(
        model=args.model or base_cfg["model"],
        api_key_env=base_cfg.get("api_key_env", "LINKAPI_API_KEY"),
        api_base=args.api_base or base_cfg.get("api_base"),
        temperature=args.temperature,
        max_tokens=cfg["inner_loop"].get("max_tokens"),
    )


def _build_mask_map(mask_root: Path) -> dict[str, Path]:
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")
    mask_map: dict[str, Path] = {}
    for path in sorted(mask_root.rglob("*_pred.png")):
        sample_id = path.name.removesuffix("_pred.png")
        if sample_id in mask_map:
            raise ValueError(
                f"Duplicate predicted mask for sample {sample_id}: {path} and {mask_map[sample_id]}"
            )
        mask_map[sample_id] = path
    return mask_map


def load_test_samples(
    test_image_list: Path,
    mask_root: Path,
    precomputed_artifact_root: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not test_image_list.is_file():
        raise FileNotFoundError(f"Test image list not found: {test_image_list}")
    with test_image_list.open("rb") as handle:
        image_paths = pickle.load(handle)
    if not isinstance(image_paths, (list, tuple)):
        raise TypeError(
            f"Expected list/tuple in {test_image_list}, got {type(image_paths).__name__}"
        )
    sample_paths = image_paths[:limit] if limit is not None else image_paths
    artifact_root = (
        precomputed_artifact_root
        if precomputed_artifact_root and precomputed_artifact_root.is_dir()
        else None
    )
    missing_artifact_ids = {
        Path(raw_path).stem
        for raw_path in sample_paths
        if artifact_root is None
        or not (artifact_root / f"{Path(raw_path).stem}.jsonl").is_file()
    }
    mask_map = _build_mask_map(mask_root) if missing_artifact_ids else {}
    samples: list[dict[str, Any]] = []
    for raw_path in sample_paths:
        image_path = Path(raw_path)
        sample_id = image_path.stem
        artifact_path = (
            artifact_root / f"{sample_id}.jsonl" if artifact_root is not None else None
        )
        if artifact_path is not None and artifact_path.is_file():
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "mask_path": None,
                    "precomputed_artifact_path": str(artifact_path),
                }
            )
            continue

        mask_path = mask_map.get(sample_id)
        if mask_path is None:
            raise FileNotFoundError(f"Missing predicted mask for sample {sample_id}")
        samples.append(
            {
                "sample_id": sample_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )
    return samples


def _is_mask_empty(mask_path: str | Path) -> bool:
    mask = load_binary_mask(mask_path)
    return not any(any(row) for row in mask)


def _gzip_output(output_path: Path) -> Path:
    gzip_path = Path(f"{output_path}.gz")
    with output_path.open("rb") as src, gzip.open(gzip_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gzip_path


def run_submission(args: argparse.Namespace) -> tuple[Path, Path]:
    cfg = load_config()
    output_path = Path(args.output)
    gzip_path = Path(f"{output_path}.gz")
    output_paths = [output_path, gzip_path]
    if all(path.exists() for path in output_paths) and not args.force:
        print(f"Already complete, skipping: {[str(path) for path in output_paths]}")
        return output_path, gzip_path

    samples = load_test_samples(
        Path(args.test_image_list),
        Path(args.mask_root),
        precomputed_artifact_root=(
            Path(args.precomputed_artifacts) if args.precomputed_artifacts else None
        ),
        limit=args.limit,
    )
    run_cfg = dict(cfg["inner_loop"])
    run_cfg["overlay_dir"] = str(ROOT / run_cfg.get("overlay_dir", "logs/overlays"))
    model_client = _make_base_client(cfg, args)
    harness = load_harness(args.harness, model_client, run_cfg)
    if args.load_memory:
        harness.set_state(Path(args.load_memory).read_text(encoding="utf-8"))

    logger = JSONLLogger(args.log)
    trace_samples, trace_meta = _load_trace_samples(args.log)
    current_model = getattr(model_client, "model", "unknown")
    can_resume = True
    if trace_meta:
        if trace_meta.get("harness") not in {None, args.harness}:
            can_resume = False
        if trace_meta.get("model") not in {None, current_model}:
            can_resume = False
    if not trace_meta or not can_resume:
        logger.log(
            "meta", harness=args.harness, model=current_model, total=len(samples)
        )
    elif trace_samples:
        print(f"Resuming from {args.log} with {len(trace_samples)} completed samples")
    elif trace_meta:
        print(f"Resuming from {args.log} with no completed sample entries found")

    def predict_one(idx: int, sample: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started = time.perf_counter()
        precomputed = None
        if sample.get("precomputed_artifact_path"):
            precomputed = _read_precomputed_artifact(
                Path(sample["precomputed_artifact_path"])
            )
        if precomputed is None:
            raise ValueError(
                f"Missing precomputed artifact for sample {sample['sample_id']}"
            )

        overlay_image_url, overlay_path = _resolve_precomputed_overlay(precomputed)
        timed_sample = {
            **sample,
            "precomputed_boxes": precomputed.get("boxes") or [],
            "precomputed_mask_empty": bool(precomputed.get("mask_empty")),
            "precomputed_overlay_image_url": overlay_image_url,
            "precomputed_overlay_path": overlay_path,
        }
        if timed_sample["precomputed_mask_empty"]:
            report = build_authentic_report(AUTHENTIC_SUMMARY)
            metadata = {"mask_empty": True}
        else:
            report, metadata = _predict_with_fallback(harness, timed_sample)
            metadata = {**metadata, "mask_empty": False}

        runtime_seconds = time.perf_counter() - started
        metadata = {
            **metadata,
        }
        return idx, {
            "sample_id": sample["sample_id"],
            "image_name": sample["image_name"],
            "report": report,
            "metadata": metadata,
            "runtime_seconds": round(runtime_seconds, 3),
        }

    results: list[dict[str, Any] | None] = [None] * len(samples)
    prefilled = 0
    if can_resume and trace_samples:
        index_by_image = {
            sample["image_name"]: idx for idx, sample in enumerate(samples)
        }
        for image_name, entry in trace_samples.items():
            idx = index_by_image.get(image_name)
            if idx is None:
                continue
            results[idx] = {
                "sample_id": samples[idx]["sample_id"],
                "image_name": image_name,
                "report": entry["report"],
                "metadata": entry.get("metadata", {}),
                "runtime_seconds": entry.get("runtime_seconds", 0.0),
            }
            prefilled += 1
    elif trace_meta and not can_resume:
        print(
            "Existing trace metadata does not match current harness/model; starting fresh"
        )

    completed = prefilled
    total_samples = len(samples)
    pending_samples = [
        (idx, sample) for idx, sample in enumerate(samples) if results[idx] is None
    ]

    if pending_samples:
        with ThreadPoolExecutor(
            max_workers=max(1, min(args.max_workers, len(pending_samples)))
        ) as executor:
            futures = {
                executor.submit(predict_one, idx, sample): idx
                for idx, sample in pending_samples
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                initial=prefilled,
                desc="Generating submission",
            ):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                logger.log("sample", **result)

        # for idx, sample in pending_samples:
        #     if results[idx] is not None:
        #         continue
        #     _, result = predict_one(idx, sample)
        #     results[idx] = result
        #     completed += 1
        #     print(f"Finished {completed}/{total_samples} samples", flush=True)
        #     logger.log("sample", **result)

    submissions = [
        {"image_name": result["image_name"], "report": result["report"]}
        for result in results
        if result is not None
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in submissions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    gzip_path = _gzip_output(output_path)
    if args.save_memory:
        save_path = Path(args.save_memory)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(harness.get_state(), encoding="utf-8")

    logger.log(
        "done",
        total=len(submissions),
        output=str(output_path),
        gzip_output=str(gzip_path),
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {gzip_path}")
    return output_path, gzip_path


def _build_parser(cfg: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RealText-V2 submission reports for the test set"
    )
    parser.add_argument(
        "--harness", default="agents/template_report_boxreasons_coordspanrepair.py"
    )
    parser.add_argument("--test-image-list", default=str(DEFAULT_TEST_IMAGE_LIST))
    parser.add_argument("--mask-root", default=str(DEFAULT_MASK_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument(
        "--temperature", type=float, default=cfg["inner_loop"].get("temperature")
    )
    parser.add_argument("--max-workers", type=int, default=int(WORKER_COUNT))
    parser.add_argument("--save-memory", default=None)
    parser.add_argument("--load-memory", default=None)
    parser.add_argument("--stub-response", default=None)
    parser.add_argument(
        "--precomputed-artifacts", default=str(DEFAULT_PRECOMPUTED_ARTIFACTS)
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    cfg = load_config()
    args = _build_parser(cfg).parse_args()
    run_submission(args)


if __name__ == "__main__":
    main()
