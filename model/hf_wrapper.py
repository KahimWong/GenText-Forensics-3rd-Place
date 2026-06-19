# ---------------------------------------------------------------
# Hugging Face Hub wrapper for EoMT model (branded as SEED).
#
# Usage:
#   # Save
#   model = EoMTForTamperingDetection()
#   model.load_state_dict_from_ckpt("path/to/checkpoint.pth")
#   model.save_pretrained("./hf_model")
#   model.push_to_hub("mps-lab/SEED")
#
#   # Load
#   model = EoMTForTamperingDetection.from_pretrained("mps-lab/SEED")
# ---------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

from model.eomt_sep_query import EoMT

logger = logging.getLogger(__name__)

# Try to import safetensors; fall back to torch.save if unavailable.
try:
    from safetensors.torch import save_file as safetensors_save_file
    from safetensors.torch import load_file as safetensors_load_file

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False
    logger.warning(
        "safetensors not installed; falling back to pytorch_model.bin. "
        "Install with: pip install safetensors"
    )


class EoMTForTamperingDetection(EoMT):
    """
    HF-compatible wrapper around EoMT that supports save_pretrained /
    from_pretrained / push_to_hub from the Hugging Face Hub.
    """

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------

    def _config(self) -> dict:
        """Export the model configuration as a JSON-serialisable dict."""
        import cfg

        lora_rank = None
        target_keywords = ("attention", "mlp")
        if self.finetune_mode == "lora":
            lora_rank = getattr(cfg, "lora_rank", 32)
            target_keywords = tuple(getattr(cfg, "moeffort_target_keywords", target_keywords))

        return {
            "model_type": "SEED",
            "backbone": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "img_size": [
                self.backbone.patch_embed.grid_size[0]
                * self.backbone.patch_embed.patch_size[0],
                self.backbone.patch_embed.grid_size[1]
                * self.backbone.patch_embed.patch_size[1],
            ],
            "num_classes": self.num_classes,
            "num_q": self.num_q,
            "num_blocks": self.num_blocks,
            "finetune_mode": self.finetune_mode,
            "lora_rank": lora_rank,
            "moeffort_target_keywords": list(target_keywords),
        }

    def save_pretrained(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the model to *save_directory* in Hugging Face format.

        Creates:
            config.json          – model configuration
            model.safetensors    – weights (or pytorch_model.bin fallback)
        """
        os.makedirs(save_directory, exist_ok=True)

        # Config
        config = self._config()
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {config_path}")

        # Weights
        state_dict = self.state_dict()
        if _HAS_SAFETENSORS:
            weight_path = os.path.join(save_directory, "model.safetensors")
            safetensors_save_file(state_dict, weight_path)
        else:
            weight_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(state_dict, weight_path)
        logger.info(f"Weights saved to {weight_path}")

        if push_to_hub:
            self.push_to_hub(save_directory, **kwargs)

    # ------------------------------------------------------------------
    #  Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "EoMTForTamperingDetection":
        """
        Instantiate an EoMTForTamperingDetection from a local directory
        or a Hugging Face Hub repo.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to a local directory containing config.json +
            model.safetensors (or pytorch_model.bin), or a HF repo id.
        """
        # Resolve local path (downloads from Hub if needed)
        if os.path.isdir(pretrained_model_name_or_path):
            model_dir = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                pretrained_model_name_or_path, **kwargs
            )

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"config.json not found in {model_dir}"
            )
        with open(config_path, "r") as f:
            config = json.load(f)

        # Filter & map config keys → EoMT.__init__ parameter names
        init_kwargs = {}
        for ck in (
            "img_size",
            "num_classes",
            "num_q",
            "num_blocks",
            "finetune_mode",
        ):
            if ck in config:
                init_kwargs[ck] = config[ck]

        if config.get("lora_rank") is not None:
            init_kwargs["lora_rank"] = config["lora_rank"]
        if config.get("target_keywords"):
            init_kwargs["target_keywords"] = tuple(config["target_keywords"])
        elif config.get("moeffort_target_keywords"):
            init_kwargs["target_keywords"] = tuple(config["moeffort_target_keywords"])

        model = cls(**init_kwargs)

        # Load weights
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.isfile(safetensors_path):
            if not _HAS_SAFETENSORS:
                raise ImportError(
                    "safetensors is required to load model.safetensors. "
                    "Install with: pip install safetensors"
                )
            state_dict = safetensors_load_file(safetensors_path)
        elif os.path.isfile(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor pytorch_model.bin found in {model_dir}"
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")

        return model

    # ------------------------------------------------------------------
    #  Convenience: load from training checkpoint
    # ------------------------------------------------------------------

    def load_state_dict_from_ckpt(
        self,
        checkpoint_path: str,
        map_location: str = "cpu",
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Load weights from a training checkpoint (.pth file containing
        {'model': state_dict, 'optimizer': ..., ...}).
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        if "model" in ckpt:
            state_dict = ckpt["model"]
            # Strip "module." prefix from DDP wrapping
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        else:
            state_dict = ckpt
        return self.load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    #  push_to_hub
    # ------------------------------------------------------------------

    def push_to_hub(
        self,
        repo_id: str,
        save_directory: Optional[str] = None,
        commit_message: str = "Upload EoMT model",
        **kwargs,
    ):
        """
        Save the model locally (if not already saved) and push to the
        Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Hugging Face repo id, e.g. "mps-lab/SEED".
        save_directory : str, optional
            Local directory to save to before pushing. If None, a
            temporary directory is used.
        commit_message : str
            Commit message for the Hub push.
        """
        from huggingface_hub import HfApi, create_repo

        if save_directory is None:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                self.save_pretrained(tmpdir)
                api = HfApi()
                create_repo(repo_id, exist_ok=True)
                api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=repo_id,
                    commit_message=commit_message,
                    **kwargs,
                )
        else:
            self.save_pretrained(save_directory)
            api = HfApi()
            create_repo(repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                commit_message=commit_message,
                **kwargs,
            )
        logger.info(f"Model pushed to https://huggingface.co/{repo_id}")


# ------------------------------------------------------------------
#  Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Instantiate fresh
    print("Creating model ...")
    model = EoMTForTamperingDetection()
    print(f"  finetune_mode = {model.finetune_mode}")
    print(f"  num_q         = {model.num_q}")
    print(f"  num_blocks    = {model.num_blocks}")

    # Load from training checkpoint (pass path as first arg)
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict_from_ckpt(ckpt_path)

        # Save locally
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "./hf_model_out"
        model.save_pretrained(out_dir)
        print(f"Saved to {out_dir}")

        # Round-trip test
        print("Round-trip load ...")
        model2 = EoMTForTamperingDetection.from_pretrained(out_dir)
        print("Round-trip OK.")
    else:
        print("Usage: python model/hf_wrapper.py <checkpoint.pth> [output_dir]")
