# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
import os
from typing import Optional

import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model.scale_block import ScaleBlock


os.environ.setdefault("HF_HOME", cfg.hf_cache_root)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cfg.hf_hub_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", cfg.hf_transformers_cache)


class EoMT(nn.Module):
    def __init__(
        self,
        img_size=(cfg.realtextv2_img_size, cfg.realtextv2_img_size),
        num_classes=1,
        num_q=cfg.num_q,
        num_blocks=4,
        finetune_mode=cfg.finetune_mode,
        lora_rank=cfg.lora_rank,
        target_keywords=cfg.moeffort_target_keywords,
    ):
        super().__init__()
        self.finetune_mode = finetune_mode

        self.backbone = self.transformers_to_timm(
            AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                cache_dir=cfg.hf_hub_cache,
            ),
            img_size,
        )

        if finetune_mode not in ("lora", "full"):
            raise ValueError(
                f"Unsupported finetune_mode: {finetune_mode}. "
                "Expected 'lora' or 'full'."
            )

        if finetune_mode == "lora":
            from model.lora import apply_lora
            self.backbone = apply_lora(
                self.backbone,
                rank=lora_rank,
                target_keywords=target_keywords,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.num_q = num_q
        self.num_classes = num_classes
        self.num_blocks = num_blocks

        dim = self.backbone.embed_dim
        patch_size = self.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q_table = nn.Embedding(self.num_q, dim)
        self.image_head = nn.Linear(dim, 2)

        self.class_head = nn.Linear(dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        self.upscale = nn.Sequential(
            *[ScaleBlock(self.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _select_query(self, batch_size: int) -> torch.Tensor:
        device = self.q_table.weight.device
        return self.q_table(
            torch.arange(self.num_q, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        )

    def transformers_to_timm(self, backbone, img_size):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        if hasattr(backbone, "layer"):
            backbone.blocks = backbone.layer
        elif hasattr(backbone, "model") and hasattr(backbone.model, "layer"):
            backbone.blocks = backbone.model.layer
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            backbone.blocks = backbone.encoder.layer
        else:
            raise AttributeError(
                f"Unsupported backbone structure for {backbone.__class__.__name__}: "
                "cannot find transformer layers at .layer, .model.layer, or .encoder.layer."
            )

        if hasattr(backbone.patch_embed, "mask_token"):
            del backbone.patch_embed.mask_token
        del backbone.embeddings
        if hasattr(backbone, "layer"):
            del backbone.layer
        if hasattr(backbone, "model"):
            del backbone.model
        if hasattr(backbone, "encoder"):
            del backbone.encoder

        return backbone

    def _predict(
        self,
        x: torch.Tensor,
        qn: int,
        return_features: bool = False,
    ):
        q = x[:, :qn, :]
        class_logits = self.class_head(q)

        x = x[:, qn + self.backbone.num_prefix_tokens:, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw",
            self.mask_head(q),
            self.upscale(x),
        )

        if return_features:
            return mask_logits, class_logits, q, x
        return mask_logits, class_logits, q

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob, qn: Optional[int] = None):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], qn, device=attn_mask.device) > prob
            )
            attn_mask[:, :qn, qn + self.backbone.num_prefix_tokens :][
                random_queries
            ] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = (
                mask[:, None, ...]
                .expand(-1, module.num_heads, -1, -1)
                .contiguous()
            )

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(
        self,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        i: int,
        qn: Optional[int] = None,
    ):
        B, N = x.shape[0], x.shape[1]
        attn_mask = torch.ones(B, N, N, dtype=torch.bool, device=x.device)

        interpolated = F.interpolate(
            mask_logits, self.backbone.patch_embed.grid_size, mode="bilinear"
        )
        interpolated = interpolated.view(
            interpolated.size(0), interpolated.size(1), -1
        )  # [B,1,P]

        patch_start = qn + self.backbone.num_prefix_tokens

        attn_mask[:, :qn, patch_start:] = interpolated > 0

        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[i - len(self.backbone.blocks) + self.num_blocks],
            qn=qn,
        )
        return attn_mask

    def _forward_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hasattr(block, "attn"):
            attn = block.attn
        else:
            attn = block.attention

        attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
        if hasattr(block, "ls1"):
            x = x + block.ls1(attn_out)
        elif hasattr(block, "layer_scale1"):
            x = x + block.layer_scale1(attn_out)

        mlp_out = block.mlp(block.norm2(x))
        if hasattr(block, "ls2"):
            x = x + block.ls2(mlp_out)
        elif hasattr(block, "layer_scale2"):
            x = x + block.layer_scale2(mlp_out)

        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        x = x / 255.0
        x = (x - self.pixel_mean) / self.pixel_std

        rope = None
        if hasattr(self.backbone, "rope_embeddings"):
            rope = self.backbone.rope_embeddings(x)

        x = self.backbone.patch_embed(x)

        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        q_sel = self._select_query(x.shape[0])
        qn = q_sel.size(1)

        for i, block in enumerate(self.backbone.blocks):
            if i == len(self.backbone.blocks) - self.num_blocks:
                x = torch.cat((q_sel, x), dim=1)

            if i >= len(self.backbone.blocks) - self.num_blocks:
                mask_logits, class_logits, _ = self._predict(
                    self.backbone.norm(x), qn=qn,
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                attn_mask = self._attn_mask(x, mask_logits, i, qn=qn)

            x = self._forward_block(block, x, attn_mask, rope)

        x = self.backbone.norm(x)
        image_logits = self.image_head(x[:, qn, :])

        if return_features:
            mask_logits, class_logits, query, feat_map = self._predict(
                x, qn=qn, return_features=True,
            )
        else:
            mask_logits, class_logits, query = self._predict(x, qn=qn)

        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        if return_features:
            return (mask_logits_per_layer, class_logits_per_layer, query, feat_map, image_logits)
        return (mask_logits_per_layer, class_logits_per_layer, query, image_logits)


if __name__ == "__main__":
    model = EoMT()
    x = torch.randn(2, 3, 512, 512)
    mask_logits_per_layer, class_logits_per_layer, query, image_logits = model(x)
    print(
        len(mask_logits_per_layer),
        len(class_logits_per_layer),
        tuple(query.shape),
        tuple(image_logits.shape),
    )
