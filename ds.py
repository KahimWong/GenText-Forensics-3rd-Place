import csv
import os
import os.path as op
import pickle
import sys
import tempfile
import zlib
from copy import deepcopy
from pathlib import Path
from random import randint

import cv2
import lmdb
import numpy as np
import six
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import cfg

# ---------------------------------------------------------------------------
#  globals
# ---------------------------------------------------------------------------

_JPEG_TMP_DIR = Path(tempfile.gettempdir()) / "realtextv2_jpeg"
_JPEG_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_JPEG_TMP_DIR)

mask_totsr = ToTensorV2()
DOC_PRO_DET_PATH_PKL_DIR = ""  # path to detection eval pkl directory

# ---------------------------------------------------------------------------
#  shared helpers
# ---------------------------------------------------------------------------

def _apply_jpeg(img, quality=100):
    """Apply a single JPEG compression round and return the RGB image."""
    img = img.copy().convert("RGB")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
        img.save(tmp.name, "JPEG", quality=quality)
        with Image.open(tmp.name) as jpeg_img:
            jpeg_img.load()
            return jpeg_img.convert("RGB")


def _configure_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _normalize_eval_mode(eval_mode=None):
    eval_mode = getattr(cfg, "eval_mode", "loc") if eval_mode is None else eval_mode
    if eval_mode == "both":
        return eval_mode
    if eval_mode not in ("loc", "det"):
        raise ValueError(f"Unsupported cfg.eval_mode: {eval_mode}")
    return eval_mode


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stable_bucket(sample_id, bucket_n):
    if bucket_n <= 0:
        raise ValueError("bucket_n must be positive")
    return zlib.crc32(sample_id.encode("utf-8")) % bucket_n


def _build_file_map(root_dir, suffixes):
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    file_map = {}
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.name in file_map:
            raise ValueError(
                f"Duplicate filename detected for {path.name}: {file_map[path.name]} and {path}"
            )
        file_map[path.name] = path
    return file_map


def _to_tensor(image):
    return mask_totsr(image=image)["image"]


def _load_binary_mask(mask_path, image_shape):
    if mask_path is None:
        return np.zeros(image_shape[:2], dtype=np.uint8)
    with Image.open(mask_path) as mask_image:
        return (np.array(mask_image.convert("L")) != 0).astype(np.uint8)


# ---------------------------------------------------------------------------
#  train-dataset helpers
# ---------------------------------------------------------------------------

def _normalize_metadata_row(row):
    return {
        "sample_id": str(row["sample_id"]),
        "language": str(row.get("language", "")),
        "language_code": str(row.get("language_code", "")),
        "type": str(row.get("type", "")),
        "image_file": str(row.get("image_file", "")),
        "mask_file": str(row.get("mask_file", "")),
        "has_mask": _as_bool(row.get("has_mask", False)),
    }


def _load_metadata_rows(data_root):
    data_root = Path(data_root)
    parquet_path = data_root / "metadata.parquet"
    if parquet_path.is_file():
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            try:
                records = pd.read_parquet(parquet_path).to_dict("records")
            except ImportError:
                pass
            else:
                return [_normalize_metadata_row(record) for record in records]

    csv_path = data_root / "metadata.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    _configure_csv_field_size_limit()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_normalize_metadata_row(row) for row in reader]


def _build_resize():
    height = int(getattr(cfg, "realtextv2_train_img_h", cfg.realtextv2_img_size))
    width = int(getattr(cfg, "realtextv2_train_img_w", cfg.realtextv2_img_size))
    return A.Resize(height=height, width=width)


# ---------------------------------------------------------------------------
#  eval-dataset helpers
# ---------------------------------------------------------------------------

def _lmdb_load_data(idx, lmdb_handle):
    img_key = "image-%09d" % idx
    img_buf = lmdb_handle.get(img_key.encode("utf-8"))
    buf = six.BytesIO()
    buf.write(img_buf)
    buf.seek(0)
    img = Image.open(buf)
    lbl_key = "label-%09d" % idx
    lbl_buf = lmdb_handle.get(lbl_key.encode("utf-8"))
    mask = (cv2.imdecode(np.frombuffer(lbl_buf, dtype=np.uint8), 0) != 0).astype(np.uint8)
    return img, mask


def _normalize_eval_mode(eval_mode=None):
    eval_mode = getattr(cfg, "eval_mode", "loc") if eval_mode is None else eval_mode
    if eval_mode == "both":
        return eval_mode
    if eval_mode not in ("loc", "det"):
        raise ValueError(f"Unsupported cfg.eval_mode: {eval_mode}")
    return eval_mode


def _get_docpro_path_pkl_dir(eval_mode=None):
    eval_mode = _normalize_eval_mode(eval_mode)
    if eval_mode == "loc":
        return cfg.path_pkl_dir
    if eval_mode == "det":
        return getattr(cfg, "det_path_pkl_dir", cfg.path_pkl_dir)
    raise ValueError(f"Unsupported cfg.eval_mode: {eval_mode}")


def _unpack_docpro_path_item(path_item):
    if len(path_item) == 3:
        return path_item[0], path_item[1], path_item[2]
    if len(path_item) == 2:
        return path_item[0], path_item[1], None
    raise ValueError(
        "Each DocPro path list item must have either 2 items "
        "(img_path, mask_path) or 3 items (img_path, mask_path, hv_mask)."
    )


# ===========================================================================
#  TRAIN DATASETS
# ===========================================================================

class RealTextV2TrainDs(Dataset):
    def __init__(self):
        self.data_root = Path(cfg.realtextv2_data_root)
        self.crop_aug = getattr(cfg, "realtextv2_train_crop_aug", None)
        self.crop_replay_aug = (
            A.ReplayCompose([deepcopy(self.crop_aug)]) if self.crop_aug is not None else None
        )
        self.resize = _build_resize()

        self.image_root = self.data_root / "train" / "image"
        self.mask_root = self.data_root / "train" / "mask"

        self.image_map = _build_file_map(self.image_root, {".jpg", ".png"})
        self.mask_map = _build_file_map(self.mask_root, {".png"})

        rows = _load_metadata_rows(self.data_root)
        self.rows = self._build_rows(rows)
        self.sample_n = len(self.rows)
        if self.sample_n == 0:
            raise ValueError("No samples found")

        self.ds_len = cfg.ds_len

    def _build_rows(self, rows):
        mapped_rows = []
        for row in rows:
            image_path = self.image_map.get(row["image_file"])
            if image_path is None:
                raise FileNotFoundError(
                    f"Image file missing for {row['sample_id']}: {row['image_file']}"
                )
            mask_path = None
            if row["mask_file"]:
                mask_path = self.mask_map.get(row["mask_file"])
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Mask file missing for {row['sample_id']}: {row['mask_file']}"
                    )
            item = dict(row)
            item["image_path"] = image_path
            item["mask_path"] = mask_path
            mapped_rows.append(item)
        return mapped_rows

    def _load_raw_sample(self, row):
        image_np = self._load_image_np(row["image_path"])
        mask = self._load_mask_np(row.get("mask_path"), image_np.shape[:2])

        if self.crop_aug is not None:
            crop_h = int(getattr(self.crop_aug, "height", image_np.shape[0]))
            crop_w = int(getattr(self.crop_aug, "width", image_np.shape[1]))
            if image_np.shape[0] >= crop_h and image_np.shape[1] >= crop_w:
                aug = self.crop_aug(image=image_np, mask=mask)
                image_np = aug["image"]
                mask = aug["mask"]

        if self.resize is not None:
            aug = self.resize(image=image_np, mask=mask)
            image_np = aug["image"]
            mask = aug["mask"]

        return Image.fromarray(image_np), mask

    def _load_image_np(self, image_path):
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))

    def _load_mask_np(self, mask_path, image_shape):
        if mask_path is not None:
            with Image.open(mask_path) as mask_image:
                return (np.array(mask_image.convert("L")) != 0).astype(np.uint8)
        return np.zeros(image_shape, dtype=np.uint8)

    def _apply_resize(self, image_np, mask):
        if self.resize is None:
            return image_np, mask
        aug = self.resize(image=image_np, mask=mask)
        return aug["image"], aug["mask"]

    def _build_item(self, row, img, mask):
        jpeg_img = _apply_jpeg(img, quality=100)
        item = {
            "img": _to_tensor(np.array(jpeg_img)).float(),
            "mask": _to_tensor(mask.copy()).long(),
            "min_qf": 100,
        }
        self._append_common_metadata(item, row)
        return item

    def _append_common_metadata(self, item, row):
        item["img_name"] = row["sample_id"]
        item["language"] = row["language"]
        item["language_code"] = row["language_code"]
        item["sample_type"] = row["type"]
        item["has_mask"] = row["has_mask"]
        item["forg_id"] = _stable_bucket(row["sample_id"], 3)
        item["split_id"] = _stable_bucket(row["sample_id"], 9)
        if "data_source" in row:
            item["data_source"] = row["data_source"]
        return item

    def __len__(self):
        return self.ds_len

    def __getitem__(self, _):
        row = self.rows[randint(0, self.sample_n - 1)]
        img, mask = self._load_raw_sample(row)
        return self._build_item(row, img, mask)


class MixedRealTextV2TrainDs(RealTextV2TrainDs):
    def __init__(self):
        super().__init__()
        self.syn_data_root = Path(cfg.realtextv2_data_root) / "syn_data"
        self.real_sample_paths_file = Path(cfg.realtextv2_data_root) / "real_sample_paths.txt"
        self.metadata_by_sample_id = {row["sample_id"]: row for row in self.rows}

        self.real_rows = [dict(row, data_source="realtextv2") for row in self.rows]
        self.real_sample_n = len(self.real_rows)

        self.clean_real_rows = self._build_clean_real_rows()
        self.clean_real_sample_n = len(self.clean_real_rows)
        if self.clean_real_sample_n == 0:
            raise ValueError("No clean real samples found")

        self.syn_rows = self._build_syn_rows()
        self.syn_sample_n = len(self.syn_rows)
        if self.syn_sample_n == 0:
            raise ValueError("No synthetic samples found")

    def _build_clean_real_rows(self):
        if not self.real_sample_paths_file.is_file():
            raise FileNotFoundError(
                f"Real sample paths file not found: {self.real_sample_paths_file}"
            )
        rows = []
        with self.real_sample_paths_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                image_path = Path(raw_line.strip())
                if not image_path:
                    continue
                if not image_path.is_file():
                    raise FileNotFoundError(f"Real sample image not found: {image_path}")
                sample_id = image_path.stem
                metadata_row = self.metadata_by_sample_id.get(sample_id, {})
                rows.append(
                    {
                        "sample_id": sample_id,
                        "language": str(metadata_row.get("language", "")),
                        "language_code": str(metadata_row.get("language_code", "")),
                        "type": str(metadata_row.get("type", "real")) or "real",
                        "has_mask": False,
                        "image_path": image_path,
                        "mask_path": None,
                        "data_source": "real_sample_paths",
                    }
                )
        return rows

    def _build_syn_rows(self):
        if not self.syn_data_root.is_dir():
            raise FileNotFoundError(
                f"Synthetic data directory not found: {self.syn_data_root}"
            )
        rows = []
        for source_dir in sorted(self.syn_data_root.iterdir()):
            if not source_dir.is_dir():
                continue
            image_dir = source_dir / "images"
            mask_dir = source_dir / "masks"
            image_map = _build_file_map(image_dir, {".jpg", ".jpeg", ".png"})
            mask_map = _build_file_map(mask_dir, {".jpg", ".jpeg", ".png"})
            for image_name, image_path in sorted(image_map.items()):
                mask_path = mask_map.get(image_name)
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Mask file missing for synthetic sample {source_dir.name}/{image_name}"
                    )
                rows.append(
                    {
                        "sample_id": f"{source_dir.name}:{image_path.stem}",
                        "language": "",
                        "language_code": "",
                        "type": source_dir.name,
                        "has_mask": True,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "data_source": "syn_data",
                    }
                )
        return rows

    def __getitem__(self, _):
        if randint(0, 1) == 0:
            row = self.real_rows[randint(0, self.real_sample_n - 1)]
        else:
            if randint(0, 1) == 0:
                row = self.clean_real_rows[randint(0, self.clean_real_sample_n - 1)]
            else:
                row = self.syn_rows[randint(0, self.syn_sample_n - 1)]
        img, mask = self._load_raw_sample(row)
        return self._build_item(row, img, mask)


class PairedMixedRealTextV2TrainDs(MixedRealTextV2TrainDs):
    def __init__(self):
        super().__init__()
        self.ds_len = max(1, self.ds_len // 2)

        self.train_real_rows = [row for row in self.real_rows if not row["has_mask"]]
        self.train_fake_rows = [row for row in self.real_rows if row["has_mask"]]
        self.train_real_sample_n = len(self.train_real_rows)
        self.train_fake_sample_n = len(self.train_fake_rows)
        if self.train_real_sample_n == 0:
            raise ValueError("No real train samples found")
        if self.train_fake_sample_n == 0:
            raise ValueError("No fake train samples found")

        self.clean_real_rows_by_name = self._build_clean_real_rows_by_name()

    def _build_clean_real_rows_by_name(self):
        rows_by_name = {}
        for row in self.clean_real_rows:
            for key in {row["image_path"].name, row["image_path"].stem, row["sample_id"]}:
                existing_row = rows_by_name.get(key)
                if existing_row is not None and existing_row["image_path"] != row["image_path"]:
                    raise ValueError(
                        f"Duplicate clean real sample mapping for {key}: "
                        f"{existing_row['image_path']} and {row['image_path']}"
                    )
                rows_by_name[key] = row
        return rows_by_name

    def _load_same_crop_pair(self, real_row, fake_row):
        fake_image_np = self._load_image_np(fake_row["image_path"])
        fake_mask = self._load_mask_np(fake_row.get("mask_path"), fake_image_np.shape[:2])
        real_image_np = self._load_image_np(real_row["image_path"])
        real_mask = self._load_mask_np(real_row.get("mask_path"), real_image_np.shape[:2])

        replay = None
        if self.crop_aug is not None:
            crop_h = int(getattr(self.crop_aug, "height", fake_image_np.shape[0]))
            crop_w = int(getattr(self.crop_aug, "width", fake_image_np.shape[1]))
            if fake_image_np.shape[0] >= crop_h and fake_image_np.shape[1] >= crop_w:
                aug = self.crop_replay_aug(image=fake_image_np, mask=fake_mask)
                fake_image_np = aug["image"]
                fake_mask = aug["mask"]
                replay = aug["replay"]
            if replay is not None:
                aug = A.ReplayCompose.replay(replay, image=real_image_np, mask=real_mask)
                real_image_np = aug["image"]
                real_mask = aug["mask"]

        real_image_np, real_mask = self._apply_resize(real_image_np, real_mask)
        fake_image_np, fake_mask = self._apply_resize(fake_image_np, fake_mask)
        return (Image.fromarray(real_image_np), real_mask), (Image.fromarray(fake_image_np), fake_mask)

    def _sample_train_pair(self):
        real_row = self.train_real_rows[randint(0, self.train_real_sample_n - 1)]
        fake_row = self.train_fake_rows[randint(0, self.train_fake_sample_n - 1)]
        return (
            (real_row, self._load_raw_sample(real_row)),
            (fake_row, self._load_raw_sample(fake_row)),
            "realtextv2",
        )

    def _resolve_clean_real_row(self, fake_image_path):
        stem = fake_image_path.stem
        stem_prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        for candidate in [
            fake_image_path.name,
            stem,
            f"{stem_prefix}{fake_image_path.suffix}",
            stem_prefix,
        ]:
            row = self.clean_real_rows_by_name.get(candidate)
            if row is not None:
                return row
        return None

    def _sample_syn_pair(self):
        fake_row = self.syn_rows[randint(0, self.syn_sample_n - 1)]
        real_row = self._resolve_clean_real_row(fake_row["image_path"])
        if real_row is None:
            raise KeyError(
                f"Unable to find clean real pair for synthetic sample {fake_row['image_path']}"
            )
        real_sample, fake_sample = self._load_same_crop_pair(real_row, fake_row)
        return (real_row, real_sample), (fake_row, fake_sample), "syn_data"

    def __getitem__(self, _):
        if randint(0, 1) == 0:
            real_item_data, fake_item_data, pair_source = self._sample_train_pair()
        else:
            real_item_data, fake_item_data, pair_source = self._sample_syn_pair()
        real_row, (real_img, real_mask) = real_item_data
        fake_row, (fake_img, fake_mask) = fake_item_data
        return {
            "real": self._build_item(real_row, real_img, real_mask),
            "fake": self._build_item(fake_row, fake_img, fake_mask),
            "pair_source": pair_source,
        }


# ===========================================================================
#  EVAL DATASETS
# ===========================================================================

class DtdValDs(Dataset):
    def __init__(self, val_name, is_sample=False, eval_mode="loc"):
        lmdb_path = op.join(cfg.data_root, f"DocTamperV1-{val_name}")
        self.lmdb = lmdb.open(
            lmdb_path, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.lmdb.begin(write=False) as txn:
            self.sample_n = int(txn.get("num-samples".encode("utf-8")))
        if is_sample:
            self.sample_n = min(cfg.val_sample_n, self.sample_n)

        self.mask_totsr = mask_totsr

        forg_type_path = op.join(cfg.forg_type_dir, f"DocTamperV1-{val_name}.pk")
        with open(forg_type_path, "rb") as f:
            self.forg_type_dict = pickle.load(f)
        self.forg_id_map = {"GE": 0, "CM": 1, "SP": 2}
        self.eval_mode = _normalize_eval_mode(eval_mode)

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as lmdb_handle:
            img_name = "%06d" % index
            img, mask = _lmdb_load_data(index, lmdb_handle)
            forg_type = self.forg_type_dict[index]
            forg_id = self.forg_id_map[forg_type]

        img = np.array(img)

        img = Image.fromarray(img)

        img = _apply_jpeg(img, quality=100)

        img = self.mask_totsr(image=np.array(img))["image"]
        ori_img = np.array(img)
        mask = self.mask_totsr(image=mask.copy())["image"]

        item = {
            "img": img,
            "mask": mask.long(),
            "img_name": img_name,
            "ori_img": ori_img,
            "forg_id": forg_id,
        }
        if self.eval_mode == "det":
            item["has_mask"] = bool(mask.any())

        return item


class DocProValDs(Dataset):
    def __init__(self, ds_name, is_sample=False, eval_mode="loc"):
        with open(op.join(_get_docpro_path_pkl_dir(eval_mode), f"{ds_name}.pkl"), "rb") as f:
            self.path_list = pickle.load(f)

        self.sample_n = (
            min(cfg.val_sample_n, len(self.path_list)) if is_sample else len(self.path_list)
        )
        self.mask_totsr = mask_totsr
        self.eval_mode = _normalize_eval_mode(eval_mode)

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        img_path, mask_path, hv_mask = _unpack_docpro_path_item(self.path_list[index])

        img_name = op.basename(img_path).split(".")[0]
        img = cv2.imread(img_path)
        mask = (cv2.imread(mask_path, 0) != 0).astype(np.uint8)

        img = cv2.resize(
            img,
            (cfg.realtextv2_img_size, cfg.realtextv2_img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (cfg.realtextv2_img_size, cfg.realtextv2_img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        img = Image.fromarray(img)
        img = _apply_jpeg(img, quality=100)

        img = self.mask_totsr(image=np.array(img))["image"]
        mask = self.mask_totsr(image=mask.copy())["image"]

        item = {
            "img": img,
            "mask": mask.long(),
            "img_name": img_name,
            "ori_img": np.array(img),
        }
        if self.eval_mode == "det":
            if hv_mask is None:
                raise ValueError(f"Expected hv_mask in det mode for sample: {img_path}")
            item["has_mask"] = bool(hv_mask)

        return item


# ===========================================================================
#  DATALOADERS
# ===========================================================================

def get_train_dl(world_size, rank, dp=False):
    ds = PairedMixedRealTextV2TrainDs()
    batch_size = max(1, cfg.train_bs // 2)
    sampler = (
        DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=True)
        if not dp
        else None
    )
    return DataLoader(
        dataset=ds, batch_size=batch_size, num_workers=cfg.dl_workers, sampler=sampler
    )


def get_val_dl(world_size, rank, dp=False, eval_mode=None):
    dl_list = {}
    eval_mode = _normalize_eval_mode(eval_mode)
    for val_name in cfg.val_name_list:
        is_sample = False
        if "sample" in val_name:
            val_name = val_name.replace("_sample", "")
            is_sample = True

        ds = (
            DtdValDs(val_name, is_sample, eval_mode=eval_mode)
            if val_name in {"FCD", "SCD", "TestingSet"}
            else DocProValDs(val_name, is_sample, eval_mode=eval_mode)
        )

        sampler = (
            DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=False)
            if not dp
            else None
        )
        dl_list[val_name] = DataLoader(
            dataset=ds, batch_size=cfg.val_bs, num_workers=cfg.dl_workers, sampler=sampler
        )
    return dl_list
