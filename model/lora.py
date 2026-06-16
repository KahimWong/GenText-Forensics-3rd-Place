import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Frozen Linear layer with a single LoRA adapter.

    y = W(x) + scale * B(A(x))

    The leading dimension (1) on lora_A / lora_B is kept for
    backward compatibility with multi-expert checkpoints.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        init_weight: torch.Tensor | None = None,
        init_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = (rank * 2 if lora_alpha is None else lora_alpha) / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if init_bias is not None:
                self.bias.data.copy_(init_bias)
        else:
            self.register_parameter("bias", None)

        # shapes (1, rank, in_features) and (1, out_features, rank)
        # — preserves ckpt compatibility with multi-expert layout
        self.lora_A = nn.Parameter(torch.empty(1, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(1, out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        y = F.linear(x2d, self.weight, self.bias)

        x2d = self.lora_dropout(x2d)
        dtype = x2d.dtype

        A = self.lora_A[0].to(dtype=dtype)
        B = self.lora_B[0].to(dtype=dtype)

        residual = F.linear(F.linear(x2d, A), B) * self.scaling
        y = y + residual

        return y.reshape(*orig_shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
#  apply helpers
# ---------------------------------------------------------------------------

def set_model_lora_expert_idx(_model: nn.Module, _expert_idx: torch.Tensor | None):
    """No-op kept for backward compat — single expert, no routing needed."""


# backward-compat aliases
set_model_pdflora_expert_idx = set_model_lora_expert_idx


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    target_keywords=("attention", "mlp"),
    lora_alpha: float | None = None,
    lora_dropout: float = 0.0,
    train_bias: bool = False,
) -> nn.Module:
    """Replace target Linear layers in the entire model with LoRA layers."""

    def _replace(lin: nn.Linear) -> nn.Module:
        return LoRALinear(
            in_features=lin.in_features,
            out_features=lin.out_features,
            rank=rank,
            bias=(lin.bias is not None),
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_weight=lin.weight.data.clone(),
            init_bias=lin.bias.data.clone() if lin.bias is not None else None,
        )

    def _rec(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in path.lower() for k in target_keywords):
                setattr(module, name, _replace(child))
            else:
                _rec(child, path)

    _rec(model)

    # freeze everything except LoRA weights
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            if train_bias and m.bias is not None:
                m.bias.requires_grad = True

    return model


# backward-compat aliases
apply_pdflora = apply_lora
apply_PDFLoRA = apply_lora
