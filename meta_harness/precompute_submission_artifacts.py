from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from llm_clients import image_to_data_url
from overlay import connected_component_boxes, load_binary_mask, render_red_box_overlay
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_IMAGE_LIST = Path("data/test_image_path_list.pkl")
DEFAULT_MASK_ROOT = Path("data/infer_path_eval")
DEFAULT_OUTPUT_DIR = Path("data/infer_path_eval/precomputed_artifacts")
DEFAULT_ARTIFACT_DIRNAME = "artifacts"
DEFAULT_WORKERS = os.cpu_count() or 1


def _build_mask_map(mask_root: Path) -> dict[str, Path]:
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")
    mask_map: dict[str, Path] = {}
    for path in sorted(mask_root.rglob("*_pred.png")):
        sample_id = path.name.removesuffix("_pred.png")
        if sample_id in mask_map:
            raise ValueError(f"Duplicate predicted mask for sample {sample_id}: {path} and {mask_map[sample_id]}")
        mask_map[sample_id] = path
    return mask_map


def load_test_samples(test_image_list: Path, mask_root: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not test_image_list.is_file():
        raise FileNotFoundError(f"Test image list not found: {test_image_list}")
    with test_image_list.open("rb") as handle:
        image_paths = pickle.load(handle)
    if not isinstance(image_paths, (list, tuple)):
        raise TypeError(f"Expected list/tuple in {test_image_list}, got {type(image_paths).__name__}")
    mask_map = _build_mask_map(mask_root)
    samples: list[dict[str, str]] = []
    for raw_path in image_paths[:limit] if limit is not None else image_paths:
        image_path = Path(raw_path)
        sample_id = image_path.stem
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


def _process_sample(task: tuple[dict[str, str], str, str]) -> str:
    sample, overlay_dir_str, artifact_dir_str = task
    overlay_dir = Path(overlay_dir_str)
    artifact_dir = Path(artifact_dir_str)
    artifact_path = artifact_dir / f"{sample['sample_id']}.jsonl"
    if artifact_path.is_file():
        return sample["sample_id"]

    mask = load_binary_mask(sample["mask_path"])
    boxes = connected_component_boxes(mask)
    mask_empty = not boxes
    overlay_path = None
    overlay_image_url = None
    if not mask_empty:
        overlay_path = render_red_box_overlay(sample["image_path"], boxes, overlay_dir / f"{sample['sample_id']}.png")
        overlay_image_url = image_to_data_url(overlay_path)
    else:
        overlay_image_url = image_to_data_url(sample["image_path"])
    row = {
        "sample_id": sample["sample_id"],
        "image_name": sample["image_name"],
        "image_path": sample["image_path"],
        "mask_path": sample["mask_path"],
        "mask_empty": mask_empty,
        "boxes": boxes,
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "overlay_image_url": overlay_image_url,
    }
    with artifact_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sample["sample_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CV artifacts for test submission")
    parser.add_argument("--test-image-list", default=str(DEFAULT_TEST_IMAGE_LIST))
    parser.add_argument("--mask-root", default=str(DEFAULT_MASK_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    artifact_dir = output_dir / DEFAULT_ARTIFACT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(Path(args.test_image_list), Path(args.mask_root), limit=args.limit)
    pending_samples = [
        sample
        for sample in samples
        if not (artifact_dir / f"{sample['sample_id']}.jsonl").is_file()
    ]

    if pending_samples:
        worker_count = max(1, args.workers)
        tasks = [(sample, str(overlay_dir), str(artifact_dir)) for sample in pending_samples]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process_sample, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Precomputing artifacts"):
                future.result()
    else:
        list(tqdm([], total=0, desc="Precomputing artifacts"))

    print(f"Wrote {artifact_dir}")


if __name__ == "__main__":
    main()
