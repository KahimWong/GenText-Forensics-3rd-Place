import os
import pickle
import shutil
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np

import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
os.environ.setdefault("HF_HOME", cfg.hf_cache_root)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cfg.hf_hub_cache)
os.environ.setdefault("HF_HUB_CACHE", cfg.hf_hub_cache)

import torch

num_gpus = torch.cuda.device_count()
torch.set_num_threads(1)

from tqdm import tqdm
from collections import defaultdict

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
import logging
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

from base_trainer import BaseTrainer
from model.eomt_sep_query import EoMT
from model.mask_classification_loss import MaskClassificationLoss
from ds import get_val_dl, get_train_dl, _load_binary_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class Trainer(BaseTrainer):
    def __init__(self, rank, world_size):
        super(Trainer, self).__init__(rank, world_size)

        # data loader
        self.train_dl = get_train_dl(self.world_size, self.rank)
        self.val_dls_map = {
            "loc": get_val_dl(self.world_size, self.rank, eval_mode="loc"),
            "det": get_val_dl(self.world_size, self.rank, eval_mode="det"),
        }
 
        # model
        self.model = EoMT().to(device=f"cuda:{self.rank}")
        self.load_ckpt(cfg.ckpt)
        self.model = DDP(
            self.model, device_ids=[self.rank], find_unused_parameters=False
        )

        # optimizer and scheduler
        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, len(self.train_dl) * cfg.epochs, eta_min=cfg.min_lr
        )

        # loss
        self.criterion = MaskClassificationLoss().to(f"cuda:{self.rank}")
        self.scaler = GradScaler()

    def _prepare_train_batch(self, items):
        device = f"cuda:{self.rank}"
        if "img" in items:
            return {
                "img": items["img"].to(device),
                "mask": items["mask"].to(device),
                "is_fake": items["has_mask"].to(device).long(),
                "min_qf": items["min_qf"],
            }

        real_items = items["real"]
        fake_items = items["fake"]
        return {
            "img": torch.cat([real_items["img"], fake_items["img"]], dim=0).to(device),
            "mask": torch.cat([real_items["mask"], fake_items["mask"]], dim=0).to(device),
            "is_fake": torch.cat([real_items["has_mask"], fake_items["has_mask"]], dim=0).to(device).long(),
            "min_qf": torch.cat([real_items["min_qf"], fake_items["min_qf"]], dim=0),
        }

    def build_targets(self, gt_masks: torch.Tensor, is_fake: torch.Tensor):
        # gt_masks: [B, 1, H, W], values in {0,1} (bool/uint8/float ok)
        # is_fake: [B], 0 for real, 1 for fake
        B, _, H, W = gt_masks.shape
        targets = []
        for b in range(B):
            if bool(is_fake[b].item()):
                targets.append(
                    {
                        "masks": gt_masks[b, 0].unsqueeze(0).float(),  # [1, H, W]
                        "labels": torch.zeros(
                            1, dtype=torch.long, device=gt_masks.device
                        ),  # [1], foreground class id 0
                    }
                )
            else:
                targets.append(
                    {
                        "masks": torch.empty(
                            0,
                            H,
                            W,
                            dtype=torch.float32,
                            device=gt_masks.device,
                        ),
                        "labels": torch.empty(
                            0,
                            dtype=torch.long,
                            device=gt_masks.device,
                        ),
                    }
                )
        return targets

    def logits_from_queries(self, mask_logits, class_logits, out_hw=None, threshold=0.5):
        # masks: [B, Q, H, W], cls: [B, Q, C+1]  (last dim includes the "no-object" class)

        mask_logits = F.interpolate(
            mask_logits, size=out_hw, mode="bilinear", align_corners=False
        )

        cls = class_logits.softmax(-1)  # [B,Q,2]
        fg_scores = cls[..., 0]  # foreground is 0 when num_labels=1
        best_q = fg_scores.argmax(dim=1)  # [B]

        # best_q = torch.zeros(
        #     mask_logits.size(0), dtype=torch.long, device=mask_logits.device
        # )  # [B], always select the first query

        mask = mask_logits.sigmoid()  # [B,Q,H,W]
        best_mask = mask[
            torch.arange(mask.size(0), device=mask.device), best_q
        ]  # [B,H,W]
        pred = (best_mask >= threshold).to(torch.uint8)

        return pred

    def _compute_binary_detection_f1(
        self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ):
        pred = pred.to(dtype=torch.float32).view(-1)
        target = target.to(dtype=torch.float32).view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        return (2.0 * tp / (2.0 * tp + fp + fn + eps)).item()

    def _resolve_model_input_hw(self):
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        patch_embed = model_ref.backbone.patch_embed
        grid_size = getattr(patch_embed, "grid_size", None)
        patch_size = getattr(patch_embed, "patch_size", None)

        if grid_size is None or patch_size is None:
            default_size = int(getattr(cfg, "realtextv2_img_size", 512))
            return default_size, default_size

        if isinstance(grid_size, int):
            grid_h = grid_w = int(grid_size)
        else:
            grid_h, grid_w = int(grid_size[0]), int(grid_size[1])

        if isinstance(patch_size, int):
            patch_h = patch_w = int(patch_size)
        else:
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])

        return grid_h * patch_h, grid_w * patch_w

    def _pad_eval_inputs(self, img, mask=None):
        target_h, target_w = self._resolve_model_input_hw()
        orig_h, orig_w = img.shape[-2:]
        if orig_h > target_h or orig_w > target_w:
            raise ValueError(
                f"Validation sample size {(orig_h, orig_w)} exceeds model input size {(target_h, target_w)}"
            )

        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        if pad_h == 0 and pad_w == 0:
            return img, mask, (0, 0, orig_h, orig_w)

        padded_img = torch.full(
            (img.shape[0], img.shape[1], target_h, target_w),
            255,
            dtype=img.dtype,
            device=img.device,
        )
        padded_img[:, :, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w] = img

        padded_mask = None
        if mask is not None:
            padded_mask = torch.zeros(
                (mask.shape[0], mask.shape[1], target_h, target_w),
                dtype=mask.dtype,
                device=mask.device,
            )
            padded_mask[:, :, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w] = mask

        return padded_img, padded_mask, (pad_top, pad_left, orig_h, orig_w)

    def _crop_padded_prediction(self, pred, crop_box):
        pad_top, pad_left, orig_h, orig_w = crop_box
        return pred[..., pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]

    def _open_large_image(self, image_path):
        Image.MAX_IMAGE_PIXELS = None
        return Image.open(image_path)

    def _infer_pred_filename(self, image_path: Path, duplicate_stems: set[str] | None = None):
        duplicate_stems = duplicate_stems or set()
        if image_path.stem not in duplicate_stems:
            return f"{image_path.stem}_pred.png"

        path_hash = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
        return f"{image_path.stem}_{path_hash}_pred.png"

    def _load_probability_map(self, image_path, image_shape=None):
        with self._open_large_image(image_path) as image:
            prob_map = np.array(image.convert("L"), dtype=np.float32) / 255.0

        if image_shape is not None and prob_map.shape != tuple(image_shape[:2]):
            raise ValueError(
                f"Probability map shape {prob_map.shape} does not match expected image shape {tuple(image_shape[:2])}"
            )

        return prob_map

    def _predict_patch_batch(self, patch_batch):
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        amp_ctx = (
            autocast("cuda", dtype=torch.float16)
            if patch_batch.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with amp_ctx:
                outputs = model_ref(patch_batch)
                mlogits_blk, clogits_blk = outputs[:2]
                image_logits = outputs[3]

        patch_fake_threshold = float(
            getattr(cfg, "infer_path_eval_patch_fake_threshold", 0.75)
        )
        mask_logits = F.interpolate(
            mlogits_blk[-1],
            size=patch_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cls = clogits_blk[-1].softmax(-1)
        fg_scores = cls[..., 0]
        best_q = fg_scores.argmax(dim=1)
        mask_prob = mask_logits.sigmoid()
        best_mask_prob = mask_prob[
            torch.arange(mask_prob.size(0), device=mask_prob.device), best_q
        ]

        image_prob = image_logits.softmax(dim=1)[:, 1]
        image_pred = (image_prob >= patch_fake_threshold).to(torch.uint8)
        best_mask_prob = best_mask_prob * image_pred.view(-1, 1, 1).to(
            best_mask_prob.dtype
        )
        return best_mask_prob, image_pred

    def infer_from_image_paths(
        self,
        image_paths: Sequence[str],
        save_dir: str,
        patch_batch_size: int = 4,
        output_names: Sequence[str] | None = None,
    ):
        if patch_batch_size <= 0:
            raise ValueError(f"patch_batch_size must be positive, got {patch_batch_size}")

        if not image_paths:
            return []

        if output_names is not None and len(output_names) != len(image_paths):
            raise ValueError(
                "output_names must have the same length as image_paths when provided"
            )

        os.makedirs(save_dir, exist_ok=True)

        input_h, input_w = self._resolve_model_input_hw()
        device = torch.device(f"cuda:{self.rank}")
        saved_paths = []
        was_training = self.model.training
        self.model.eval()

        def compute_starts(length, window):
            if length <= window:
                return [0]

            starts = list(range(0, length - window + 1, window))
            last_start = length - window
            if starts[-1] != last_start:
                starts.append(last_start)
            return starts

        def flush_batch(batch_patches, batch_meta, merged_prob_map):
            if not batch_patches:
                return

            patch_tensor = (
                torch.from_numpy(np.stack(batch_patches, axis=0))
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .to(device)
            )
            probs, image_pred = self._predict_patch_batch(patch_tensor)
            probs_np = probs.detach().cpu().numpy().astype(np.float32)
            image_pred_np = image_pred.detach().cpu().numpy().astype(np.uint8)

            for prob_patch, patch_is_fake, meta in zip(probs_np, image_pred_np, batch_meta):
                if patch_is_fake == 0:
                    continue
                top = meta["top"]
                left = meta["left"]
                patch_h = meta["patch_h"]
                patch_w = meta["patch_w"]
                pad_top = meta["pad_top"]
                pad_left = meta["pad_left"]
                cropped_prob = prob_patch[
                    pad_top : pad_top + patch_h,
                    pad_left : pad_left + patch_w,
                ]
                current = merged_prob_map[top : top + patch_h, left : left + patch_w]
                np.maximum(current, cropped_prob, out=current)

            batch_patches.clear()
            batch_meta.clear()

        try:
            for sample_idx, image_path in enumerate(image_paths):
                image_path = Path(image_path)
                with self._open_large_image(image_path) as image:
                    orig_w, orig_h = image.size
                    rgb_image = image.convert("RGB")

                merged_prob_map = np.zeros((orig_h, orig_w), dtype=np.float32)
                batch_patches = []
                batch_meta = []

                for top in compute_starts(orig_h, input_h):
                    for left in compute_starts(orig_w, input_w):
                        bottom = min(top + input_h, orig_h)
                        right = min(left + input_w, orig_w)
                        patch = np.array(
                            rgb_image.crop((left, top, right, bottom)),
                            dtype=np.uint8,
                        )
                        patch_h = bottom - top
                        patch_w = right - left

                        pad_h = max(input_h - patch_h, 0)
                        pad_w = max(input_w - patch_w, 0)
                        pad_top = pad_h // 2
                        pad_left = pad_w // 2

                        if pad_h > 0 or pad_w > 0:
                            padded_patch = np.full((input_h, input_w, 3), 255, dtype=np.uint8)
                            padded_patch[
                                pad_top : pad_top + patch_h,
                                pad_left : pad_left + patch_w,
                            ] = patch
                        else:
                            padded_patch = patch

                        batch_patches.append(padded_patch)
                        batch_meta.append(
                            {
                                "top": top,
                                "left": left,
                                "patch_h": patch_h,
                                "patch_w": patch_w,
                                "pad_top": pad_top,
                                "pad_left": pad_left,
                            }
                        )

                        if len(batch_patches) == patch_batch_size:
                            flush_batch(batch_patches, batch_meta, merged_prob_map)

                flush_batch(batch_patches, batch_meta, merged_prob_map)

                out_name = (
                    output_names[sample_idx]  
                    if output_names is not None
                    else f"{image_path.stem}_pred.png"
                )
                out_path = Path(save_dir) / out_name
                prob_image = np.clip(np.rint(merged_prob_map * 255.0), 0, 255).astype(
                    np.uint8
                )
                Image.fromarray(prob_image, mode="L").save(out_path)
                saved_paths.append(str(out_path))
        finally:
            if was_training:
                self.model.train()

        return saved_paths
    
    def _compute_binary_confusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        pred = pred.to(dtype=torch.long).view(-1)
        target = target.to(dtype=torch.long).view(-1)
        tp = ((pred == 1) & (target == 1)).sum()
        fp = ((pred == 1) & (target == 0)).sum()
        tn = ((pred == 0) & (target == 0)).sum()
        fn = ((pred == 0) & (target == 1)).sum()
        return tp, fp, tn, fn

    def infer_path_eval(self):
        pkl_path = Path(cfg.infer_path_eval_pkl)
        with pkl_path.open("rb") as handle:
            path_items = pickle.load(handle)

        if not isinstance(path_items, list) or len(path_items) == 0:
            raise ValueError(f"Expected a non-empty list in {pkl_path}, got {type(path_items).__name__}")

        samples = []
        stem_counts = defaultdict(int)
        has_any_gt = False
        for sample_idx, item in enumerate(path_items):
            if isinstance(item, (str, Path)):
                image_path = Path(item)
                mask_path = None
            elif isinstance(item, (list, tuple)):
                if len(item) == 0:
                    raise ValueError("Encountered an empty item in the inference pickle list")
                image_path = Path(item[0])
                mask_path = None if len(item) < 2 or item[1] is None else Path(item[1])
            else:
                raise ValueError(
                    "Each pickle item must be an image path string/path or a tuple/list like (img_path, mask_path)"
                )

            if not image_path.is_file():
                raise FileNotFoundError(f"Image path not found: {image_path}")
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError(f"Mask path not found: {mask_path}")
            if mask_path is not None:
                has_any_gt = True

            stem_counts[image_path.stem] += 1

            samples.append(
                {
                    "sample_idx": sample_idx,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

        duplicate_stems = {stem for stem, count in stem_counts.items() if count > 1}
        for sample in samples:
            sample["pred_filename"] = self._infer_pred_filename(
                sample["image_path"],
                duplicate_stems=duplicate_stems,
            )

        save_root = Path(cfg.infer_path_eval_save_dir) / pkl_path.stem
        if self.rank == 0 and save_root.exists() and cfg.infer_path_eval_overwrite:
            shutil.rmtree(save_root)
        dist.barrier()

        save_dir = save_root / f"rank_{self.rank:02d}"
        if save_dir.exists() and cfg.infer_path_eval_overwrite:
            shutil.rmtree(save_dir)

        rank_samples = samples[self.rank :: self.world_size]
        image_paths = [str(sample["image_path"]) for sample in rank_samples]
        output_names = [sample["pred_filename"] for sample in rank_samples]
        self.infer_from_image_paths(
            image_paths=image_paths,
            save_dir=str(save_dir),
            patch_batch_size=int(cfg.infer_path_eval_patch_batch_size),
            output_names=output_names,
        )
        dist.barrier()

        if not has_any_gt:
            saved_count_tensor = torch.tensor(
                [float(len(rank_samples))],
                device=f"cuda:{self.rank}",
                dtype=torch.float64,
            )
            dist.all_reduce(saved_count_tensor, op=dist.ReduceOp.SUM)
            saved_count = int(saved_count_tensor.item())
            if self.rank == 0:
                logging.info(
                    "Infer path eval %s saved prediction masks for %d samples to %s",
                    pkl_path.stem,
                    saved_count,
                    save_root,
                )
            return {
                "saved_count": saved_count,
                "save_dir": str(save_root),
            }

        pixel_tp = 0.0
        pixel_fp = 0.0
        pixel_fn = 0.0
        pixel_union = 0.0
        forged_count = 0
        det_tp = 0.0
        det_fp = 0.0
        det_tn = 0.0
        det_fn = 0.0
        clean_fp = 0.0
        clean_count = 0

        for sample in tqdm(rank_samples, desc=f"infer_path_eval_rank{self.rank}"):
            image_path = sample["image_path"]
            mask_path = sample["mask_path"]
            pred_path = save_dir / sample["pred_filename"]
            if not pred_path.is_file():
                raise FileNotFoundError(f"Predicted mask not found: {pred_path}")

            with self._open_large_image(image_path) as image:
                image_shape = (image.size[1], image.size[0])
            pred_prob = self._load_probability_map(pred_path, image_shape)
            pred_mask = (
                pred_prob
                >= float(getattr(cfg, "infer_path_eval_patch_fake_threshold", 0.75))
            ).astype(np.uint8)
            gt_mask = _load_binary_mask(mask_path, image_shape)

            pred_mask = pred_mask.astype(np.uint8)
            gt_mask = gt_mask.astype(np.uint8)
            pred_has_mask = int(pred_mask.any())
            gt_has_mask = int(mask_path is not None)

            if pred_has_mask and gt_has_mask:
                det_tp += 1.0
            elif pred_has_mask and not gt_has_mask:
                det_fp += 1.0
            elif (not pred_has_mask) and gt_has_mask:
                det_fn += 1.0
            else:
                det_tn += 1.0

            if gt_has_mask:
                forged_count += 1
                matched = float((pred_mask * gt_mask).sum())
                pred_sum = float(pred_mask.sum())
                gt_sum = float(gt_mask.sum())
                pixel_tp += matched
                pixel_fp += pred_sum - matched
                pixel_fn += gt_sum - matched
                pixel_union += float(((pred_mask + gt_mask) > 0).sum())
            else:
                clean_count += 1
                if pred_has_mask:
                    clean_fp += 1.0

        metric_tensor = torch.tensor(
            [
                pixel_tp,
                pixel_fp,
                pixel_fn,
                pixel_union,
                float(forged_count),
                det_tp,
                det_fp,
                det_tn,
                det_fn,
                clean_fp,
                float(clean_count),
            ],
            device=f"cuda:{self.rank}",
            dtype=torch.float64,
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        (
            pixel_tp,
            pixel_fp,
            pixel_fn,
            pixel_union,
            forged_count,
            det_tp,
            det_fp,
            det_tn,
            det_fn,
            clean_fp,
            clean_count,
        ) = metric_tensor.tolist()

        pixel_precision = pixel_tp / (pixel_tp + pixel_fp + 1e-8) if forged_count > 0 else 0.0
        pixel_recall = pixel_tp / (pixel_tp + pixel_fn + 1e-8) if forged_count > 0 else 0.0
        pixel_f1 = (
            2.0 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-8)
            if forged_count > 0
            else 0.0
        )
        pixel_iou = pixel_tp / (pixel_union + 1e-8) if forged_count > 0 else 0.0

        total = det_tp + det_fp + det_tn + det_fn
        image_acc = (det_tp + det_tn) / (total + 1e-8) if total > 0 else 0.0
        image_precision = det_tp / (det_tp + det_fp + 1e-8) if (det_tp + det_fp) > 0 else 0.0
        image_recall = det_tp / (det_tp + det_fn + 1e-8) if (det_tp + det_fn) > 0 else 0.0
        image_f1 = (
            2.0 * image_precision * image_recall / (image_precision + image_recall + 1e-8)
            if (image_precision + image_recall) > 0
            else 0.0
        )
        clean_fpr = clean_fp / (clean_count + 1e-8) if clean_count > 0 else 0.0

        if self.rank == 0:
            logging.info(
                "Infer path eval %s pixel_f1: %.4f | pixel_precision: %.4f | "
                "pixel_recall: %.4f | pixel_iou: %.4f | image_acc: %.4f | "
                "image_precision: %.4f | image_recall: %.4f | image_f1: %.4f | "
                "clean_fpr: %.4f | tp: %d | fp: %d | tn: %d | fn: %d",
                pkl_path.stem,
                pixel_f1,
                pixel_precision,
                pixel_recall,
                pixel_iou,
                image_acc,
                image_precision,
                image_recall,
                image_f1,
                clean_fpr,
                int(det_tp),
                int(det_fp),
                int(det_tn),
                int(det_fn),
            )
        return {
            "pixel_f1": pixel_f1,
            "pixel_precision": pixel_precision,
            "pixel_recall": pixel_recall,
            "pixel_iou": pixel_iou,
            "image_acc": image_acc,
            "image_precision": image_precision,
            "image_recall": image_recall,
            "image_f1": image_f1,
            "clean_fpr": clean_fpr,
            "save_dir": str(save_root),
        }

    def evaluate(self):
        eval_modes = self._resolve_eval_modes()
        scores = {}
        if "loc" in eval_modes:
            scores["loc"] = self.val(self.val_dls_map["loc"])
        if "det" in eval_modes:
            scores["det"] = self.val_det(self.val_dls_map["det"])
        return self._select_eval_score(scores)

    def _resolve_eval_modes(self):
        eval_mode_cfg = getattr(cfg, "eval_mode", "loc")
        if isinstance(eval_mode_cfg, str):
            if eval_mode_cfg == "both":
                return ["loc", "det"]
            if eval_mode_cfg in ("loc", "det"):
                return [eval_mode_cfg]
        elif isinstance(eval_mode_cfg, (list, tuple, set)):
            modes = []
            for mode in eval_mode_cfg:
                if mode == "both":
                    modes.extend(["loc", "det"])
                elif mode in ("loc", "det"):
                    modes.append(mode)
                else:
                    raise ValueError(f"Unsupported cfg.eval_mode entry: {mode}")
            dedup_modes = []
            for mode in modes:
                if mode not in dedup_modes:
                    dedup_modes.append(mode)
            if dedup_modes:
                return dedup_modes
        raise ValueError(f"Unsupported cfg.eval_mode: {eval_mode_cfg}")

    def _select_eval_score(self, scores):
        if not scores:
            raise ValueError("No evaluation scores were produced.")
        if len(scores) == 1:
            return next(iter(scores.values()))

        score_mode = getattr(cfg, "eval_score_mode", "mean")
        if score_mode in scores:
            return scores[score_mode]
        if score_mode == "mean":
            return sum(scores.values()) / len(scores)
        raise ValueError(
            f"Unsupported eval_score_mode: {score_mode}. "
            f"Expected one of {list(scores.keys()) + ['mean']}"
        )

    def train(self):
        step = 1
        self.model.train()

        for epoch in range(1, cfg.epochs + 1):
            losses_record = defaultdict(AverageMeter)

            if epoch != 1:
                if hasattr(self.train_dl.dataset, "S"):
                    setattr(
                        self.train_dl.dataset,
                        "S",
                        self.train_dl.dataset.S + cfg.step_per_epoch,
                    )

            sampler_set_epoch = getattr(self.train_dl.sampler, "set_epoch", None)
            if callable(sampler_set_epoch):
                sampler_set_epoch(epoch)

            tqdm_fn = tqdm if self.rank == 0 else lambda x: x

            for items in tqdm_fn(self.train_dl):
                batch = self._prepare_train_batch(items)
                img = batch["img"]
                mask = batch["mask"]
                is_fake = batch["is_fake"]
                min_qf = batch["min_qf"]
                targets = self.build_targets(mask, is_fake)

                with autocast("cuda", dtype=torch.float16):
                    outputs = self.model(img)
                    mlogits_blk, clogits_blk = outputs[:2]
                    image_logits = outputs[3]

                losses_all_blocks = {}
                for i, (mask_logits, class_logits) in enumerate(
                    list(zip(mlogits_blk, clogits_blk))
                ):
                    losses = self.criterion(
                        masks_queries_logits=mask_logits.float(),
                        class_queries_logits=class_logits.float(),
                        targets=targets,
                    )
                    losses = {f"{key}{i}": value for key, value in losses.items()}
                    losses_all_blocks |= losses

                total_loss = self.criterion.loss_total(losses_all_blocks)
                image_cls_loss = F.cross_entropy(image_logits.float(), is_fake)
                total_loss = total_loss + image_cls_loss * float(
                    getattr(cfg, "img_cls_loss_weight", 1.0)
                )

                self.scaler.scale(total_loss / cfg.accum_step).backward()

                if step % cfg.accum_step == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = self.logits_from_queries(
                        mlogits_blk[-1],
                        clogits_blk[-1],
                        out_hw=mask.shape[-2:],
                    )
                    f1, _, _ = self.compute_f1(preds, mask, is_pred=True)
                    image_pred = image_logits.argmax(dim=1)
                    image_acc = (image_pred == is_fake).float().mean()
                    image_f1 = self._compute_binary_detection_f1(image_pred, is_fake)

                losses = {
                    "total_loss": total_loss.item(),
                    "image_cls_loss": image_cls_loss.item(),
                    "image_acc": image_acc.item(),
                    "image_f1": image_f1,
                    "f1": f1,
                    "min_qf": min_qf[0].item(),
                }

                for name, loss in losses.items():
                    val_tensor = torch.tensor(loss).to(f"cuda:{self.rank}")
                    dist.reduce(val_tensor, dst=0, op=dist.ReduceOp.SUM)
                    if self.rank == 0:
                        avg_val = val_tensor.item() / self.world_size
                        losses_record[name].update(avg_val)

                if self.rank == 0:
                    self.write_log(step, losses_record)
                    if step % cfg.print_log_step == 0:
                        self.print_log(step, losses_record)

                is_accum_boundary = step % cfg.accum_step == 0
                should_run_val = (step % cfg.val_step == 0) or (
                    cfg.check_val and is_accum_boundary
                )
                if should_run_val:
                    if cfg.skip_val == True:
                        score = 0.0
                    else:
                        score = self.evaluate()
                    self.model.train()
                    if self.rank == 0:
                        self.save_ckpt(step, score)
                step += 1

                self.scheduler.step()

    def val(self, val_dls=None):
        self.model.eval()
        val_dls = self.val_dls_map["loc"] if val_dls is None else val_dls

        per_ds_stats = {}

        for ds_name, val_dl in val_dls.items():
            f1_sum = 0.0
            f1_count = 0
            tqdm_fn = tqdm if self.rank == 0 else lambda x: x

            for items in tqdm_fn(val_dl):
                img = items["img"].to(f"cuda:{self.rank}")
                mask = items["mask"].to(f"cuda:{self.rank}")
                img, padded_mask, crop_box = self._pad_eval_inputs(img, mask=mask)

                with torch.no_grad():
                    with autocast("cuda", dtype=torch.float16):
                        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
                        outputs = model_ref(img)
                        mlogits_blk, clogits_blk = outputs[:2]

                preds = self.logits_from_queries(
                    mlogits_blk[-1], clogits_blk[-1], out_hw=mask.shape[-2:]
                )
                preds = self._crop_padded_prediction(preds, crop_box)
                f1, _, _ = self.compute_f1(preds, mask, is_pred=True)
                f1_sum += f1 * img.size(0)
                f1_count += img.size(0)

            stats_tensor = torch.tensor(
                [f1_sum, float(f1_count)],
                device=f"cuda:{self.rank}",
                dtype=torch.float64,
            )
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            per_ds_stats[ds_name] = (
                (stats_tensor[0] / stats_tensor[1]).item()
                if stats_tensor[1] > 0
                else 0.0
            )

        if self.rank == 0:
            for ds_name, f1_val in per_ds_stats.items():
                logging.info(f"Val {ds_name} F1: {f1_val:.4f}")

        avg_f1 = sum(per_ds_stats.values()) / len(per_ds_stats)
        self.model.train()
        return avg_f1

    def val_det(self, val_dls=None):
        self.model.eval()
        val_dls = self.val_dls_map["det"] if val_dls is None else val_dls

        per_ds_stats = {}

        for ds_name, val_dl in val_dls.items():
            tp = torch.tensor(0.0, device=f"cuda:{self.rank}")
            fp = torch.tensor(0.0, device=f"cuda:{self.rank}")
            tn = torch.tensor(0.0, device=f"cuda:{self.rank}")
            fn = torch.tensor(0.0, device=f"cuda:{self.rank}")
            tqdm_fn = tqdm if self.rank == 0 else lambda x: x

            for items in tqdm_fn(val_dl):
                img = items["img"].to(f"cuda:{self.rank}")
                is_fake = items["has_mask"].to(f"cuda:{self.rank}").long()
                img, _, crop_box = self._pad_eval_inputs(img)

                with torch.no_grad():
                    with autocast("cuda", dtype=torch.float16):
                        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
                        outputs = model_ref(img)
                        image_logits = outputs[3]

                threshold = float(getattr(cfg, "infer_path_eval_patch_fake_threshold", 0.75))
                image_probs = image_logits.softmax(dim=1)[:, 1]
                image_pred = (image_probs >= threshold).to(torch.long)

                batch_tp, batch_fp, batch_tn, batch_fn = self._compute_binary_confusion(
                    image_pred, is_fake,
                )
                tp += batch_tp.float()
                fp += batch_fp.float()
                tn += batch_tn.float()
                fn += batch_fn.float()

            stats_tensor = torch.stack([tp, fp, tn, fn])
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            per_ds_stats[ds_name] = {
                "tp": stats_tensor[0].item(),
                "fp": stats_tensor[1].item(),
                "tn": stats_tensor[2].item(),
                "fn": stats_tensor[3].item(),
            }

        per_ds_reduced = {}
        eps = 1e-8
        for ds_name, stats in per_ds_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            tn = stats["tn"]
            fn = stats["fn"]
            total = tp + fp + tn + fn
            acc = (tp + tn) / (total + eps) if total > 0 else 0.0
            precision = tp / (tp + fp + eps) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn + eps) if tp + fn > 0 else 0.0
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps) if tp + fp + fn > 0 else 0.0
            per_ds_reduced[ds_name] = {
                "image_acc": acc,
                "image_precision": precision,
                "image_recall": recall,
                "image_f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }

        if self.rank == 0:
            for ds_name, stats in per_ds_reduced.items():
                logging.info(
                    f"Val {ds_name} det acc: {stats['image_acc']:.4f} | "
                    f"det precision: {stats['image_precision']:.4f} | "
                    f"det recall: {stats['image_recall']:.4f} | "
                    f"det F1: {stats['image_f1']:.4f} | "
                    f"tp: {stats['tp']} | fp: {stats['fp']} | "
                    f"tn: {stats['tn']} | fn: {stats['fn']}"
                )

        avg_image_f1 = sum(
            stats["image_f1"] for stats in per_ds_reduced.values()
        ) / len(per_ds_reduced)
        self.model.train()
        return avg_image_f1



def main(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    trainer = Trainer(rank, world_size)
    if cfg.mode == "train":
        trainer.train()
    elif cfg.mode == "val":
        trainer.evaluate()
    elif cfg.mode == "infer_path_eval":
        trainer.infer_path_eval()
    else:
        raise ValueError(f"Unsupported cfg.mode: {cfg.mode}")


if __name__ == "__main__":
    world_size_ = torch.cuda.device_count()
    from torch.multiprocessing.spawn import spawn

    spawn(main, args=(world_size_,), nprocs=world_size_, join=True)
