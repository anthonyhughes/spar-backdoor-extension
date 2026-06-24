"""Magnitude-based unstructured pruning strategies."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


def _bake_existing_masks(model: nn.Module) -> None:
    """Remove any existing pruning hooks, making current masks permanent.

    After this call, the pruned-zero weights are baked into ``module.weight``
    and there are no longer ``weight_orig`` / ``weight_mask`` buffers.
    Subsequent pruning calls start from the current (partially-zeroed) state.
    """

    import torch.nn as nn
    import torch.nn.utils.prune as prune

    for module in model.modules():
        if isinstance(module, nn.Linear) and prune.is_pruned(module):
            prune.remove(module, "weight")


def _linear_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Return (module, "weight") pairs for all ``nn.Linear`` layers."""

    import torch.nn as nn

    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]


def _quantile_via_kthvalue(tensor: torch.Tensor, q: float) -> float:
    """Compute the q-th quantile using ``kthvalue`` (no element-count limit).

    ``torch.quantile`` caps at 2**24 elements; ``kthvalue`` does not.
    """

    import torch

    flat = tensor.flatten()
    k = max(1, round(q * flat.numel()))

    return torch.kthvalue(flat, k).values.item()


def _global_magnitude_threshold(
    params: list[tuple[nn.Module, str]], sparsity: float
) -> float:
    """Global L1-magnitude pruning threshold via a streaming histogram.

    Materialising every weight magnitude is O(total params) — ~225 GB of float32
    for 70B's MLP, and ``kthvalue`` then allocates a same-size int64 index buffer
    on top (~450 GB), enough to OOM-kill even a 755 GB host (the silent death seen
    at 70B). Instead we accumulate a fixed-bin histogram of magnitudes in two
    cheap streaming passes (O(bins) memory, ~8 MB) and read the sparsity-quantile
    off its CDF. Reductions run on each weight's own device, so there is no large
    host transfer and no full-size GPU temporary.
    """

    import torch

    if sparsity <= 0.0:
        return 0.0

    bins = 1_000_000

    # Pass 1: global max |w| + element count. max(|w|)=max(w.max,-w.min) avoids
    # an abs() temporary; both are scalar reductions.
    gmax = 0.0
    total = 0
    for m, n in params:
        w = getattr(m, n).data
        gmax = max(gmax, w.max().item(), -w.min().item())
        total += w.numel()

    if gmax <= 0.0 or total == 0:
        return 0.0

    # Pass 2: histogram of |w| over [0, gmax], accumulated on CPU (float64 so the
    # 56e9-element counts stay exact through cumsum).
    hist = torch.zeros(bins, dtype=torch.float64)
    for m, n in params:
        w = getattr(m, n).data
        h = torch.histc(w.detach().abs().float(), bins=bins, min=0.0, max=gmax)
        hist += h.double().cpu()

    # Threshold = upper edge of the bin where the CDF first reaches sparsity·total.
    cdf = torch.cumsum(hist, dim=0)
    target = torch.tensor(sparsity * total, dtype=torch.float64)
    idx = min(int(torch.searchsorted(cdf, target).item()), bins - 1)

    return (idx + 1) * (gmax / bins)


def _zero_by_threshold(params: list[tuple[nn.Module, str]], threshold: float) -> None:
    """Zero weights with magnitude at or below ``threshold``, in-place."""

    for module, name in params:
        w = getattr(module, name).data
        w[w.abs() <= threshold] = 0


# TODO: allow setting to any constant, or doing random perturbation, or some other perturbation strategy
@dataclass(frozen=True)
class GlobalMagnitudePruning:
    """Global unstructured L1-magnitude pruning across all ``nn.Linear`` layers.

    At each call the ``sparsity`` fraction of the smallest-magnitude weights
    (by absolute value, across the entire model) are zeroed.  In cumulative
    mode the already-zero weights (magnitude 0) are always included in the
    pruned set — guaranteeing that ``sparsity`` is the total fraction of
    zeros after each call.

    Threshold computation uses a streaming histogram (see
    ``_global_magnitude_threshold``) so it scales to 70B without materialising
    every weight magnitude.
    """

    @property
    def name(self) -> str:
        return "global_magnitude"

    def prune(
        self, model: nn.Module, sparsity: float, *, tokenizer: Any = None
    ) -> nn.Module:
        """Apply global L1-magnitude pruning to the given sparsity level."""

        _bake_existing_masks(model)

        params = _linear_params(model)
        threshold = _global_magnitude_threshold(params, sparsity)
        _zero_by_threshold(params, threshold)

        return model

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model


# TODO: doesn't this not discriminate between attn and mlp
@dataclass(frozen=True)
class LayerWiseMagnitudePruning:
    """Per-layer L1 unstructured pruning: each ``nn.Linear`` pruned independently.

    Each layer reaches the target ``sparsity`` independently, so the global
    sparsity equals the per-layer sparsity (assuming uniform layer sizes).
    """

    @property
    def name(self) -> str:
        return "layer_wise_magnitude"

    def prune(
        self, model: nn.Module, sparsity: float, *, tokenizer: Any = None
    ) -> nn.Module:
        """Apply per-layer L1-magnitude pruning to the given sparsity level."""

        import torch.nn as nn

        _bake_existing_masks(model)

        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue

            w = module.weight.data
            threshold = _quantile_via_kthvalue(w.abs().flatten().float(), sparsity)
            w[w.abs() <= threshold] = 0

        return model

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model


@dataclass
class TargetedLayerPruning:
    """Prune only specific layers matching a regex pattern or list of indices.

    Useful for testing whether pruning *only* the layers targeted by LoRA
    fine-tuning is sufficient to remove the backdoor.

    Args:
        layer_pattern: Regex matched against each module's full name
            (e.g. ``".*mlp.*"`` prunes all MLP layers).
        layer_indices: If non-empty, prune only Linear layers at these
            0-based indices in the flat enumeration of all ``nn.Linear``
            modules (overrides ``layer_pattern``).
    """

    layer_pattern: str = ".*"
    layer_indices: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        tag = self.layer_pattern.replace(".*", "").replace(".", "_").strip("_") or "all"

        return f"targeted_{tag}"

    def _target_params(self, model: nn.Module) -> list[tuple[nn.Module, str]]:
        """Return (module, "weight") pairs for layers matching the filter."""

        import torch.nn as nn

        named_linears = [
            (name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)
        ]

        if self.layer_indices:
            return [
                (m, "weight")
                for i, (_, m) in enumerate(named_linears)
                if i in self.layer_indices
            ]

        pattern = re.compile(self.layer_pattern)

        return [(m, "weight") for name, m in named_linears if pattern.search(name)]

    def prune(
        self, model: nn.Module, sparsity: float, *, tokenizer: Any = None
    ) -> nn.Module:
        """Apply targeted L1-magnitude pruning to matching layers."""

        _bake_existing_masks(model)
        params = self._target_params(model)

        if not params:
            return model

        threshold = _global_magnitude_threshold(params, sparsity)
        _zero_by_threshold(params, threshold)

        return model

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model


# ------------------------------------------------------------------ #
# Composable magnitude pruning                                        #
# ------------------------------------------------------------------ #


def _prune_rows(module: nn.Module, start: int, end: int, sparsity: float) -> None:
    """Prune rows ``[start:end]`` of ``module.weight`` to target sparsity."""

    w = module.weight.data[start:end]  # type: ignore[union-attr]
    threshold = _quantile_via_kthvalue(w.abs().flatten().float(), sparsity)
    w[w.abs() <= threshold] = 0


def _prune_cols(module: nn.Module, start: int, end: int, sparsity: float) -> None:
    """Prune columns ``[start:end]`` of ``module.weight`` to target sparsity."""

    w = module.weight.data[:, start:end]  # type: ignore[union-attr]
    threshold = _quantile_via_kthvalue(w.abs().flatten().float(), sparsity)
    w[w.abs() <= threshold] = 0


def _prune_attn_per_head(model: nn.Module, sparsity: float) -> None:
    """Prune each attention head's weight slices independently to target sparsity.

    For Q and O projections, each *query* head is an independent unit.
    For K and V projections, each *KV* head is an independent unit (GQA-aware).
    """

    from .heads import _discover_attention_heads

    head_infos = _discover_attention_heads(model)

    for info in head_infos:
        hd = info.head_dim

        for h in range(info.num_heads):
            start, end = h * hd, (h + 1) * hd
            _prune_rows(info.q_proj, start, end, sparsity)
            _prune_cols(info.o_proj, start, end, sparsity)

        for h in range(info.num_kv_heads):
            start, end = h * hd, (h + 1) * hd
            _prune_rows(info.k_proj, start, end, sparsity)
            _prune_rows(info.v_proj, start, end, sparsity)


@dataclass
class MagnitudePruning:
    """Composable L1-magnitude pruning across scope, component, and attention granularity.

    This single strategy subsumes :class:`GlobalMagnitudePruning`,
    :class:`LayerWiseMagnitudePruning`, and :class:`TargetedLayerPruning` by
    exposing three orthogonal axes:

    * **scope** — ``"global"`` (one threshold across all targeted params) or
      ``"layer"`` (each ``nn.Linear`` pruned independently).
    * **components** — ``"both"`` (attention + MLP layers, excluding ``lm_head``
      and other non-transformer Linear layers), ``"attn"`` (attention
      projections only), or ``"mlp"`` (MLP layers only).
    * **attn_granularity** — ``"matrix"`` (prune the full weight matrix) or
      ``"head"`` (reshape into per-head slices and prune each head
      independently).  Ignored when ``components="mlp"``.

    When ``components="both"`` and ``attn_granularity="head"``, MLP layers are
    pruned according to ``scope`` while attention layers are pruned per-head.

    Args:
        scope: Threshold computation scope.
        components: Which transformer sub-components to target.
        attn_granularity: How to partition attention weight matrices for pruning.
        attn_pattern: Regex matched against full module names to identify
            attention projections.
        mlp_pattern: Regex matched against full module names to identify MLP
            layers.
    """

    scope: str = "global"
    components: str = "both"
    attn_granularity: str = "matrix"
    attn_pattern: str = r"self_attn|attention"
    mlp_pattern: str = r"\.mlp\."

    def __post_init__(self) -> None:
        if self.scope not in ("global", "layer"):
            raise ValueError(f"scope must be 'global' or 'layer', got {self.scope!r}")
        if self.components not in ("both", "attn", "mlp"):
            raise ValueError(
                f"components must be 'both', 'attn', or 'mlp', got {self.components!r}"
            )
        if self.attn_granularity not in ("matrix", "head"):
            raise ValueError(
                f"attn_granularity must be 'matrix' or 'head', got {self.attn_granularity!r}"
            )

    @property
    def name(self) -> str:
        parts = ["magnitude", self.scope, self.components]

        if self.components != "mlp" and self.attn_granularity == "head":
            parts.append("perhead")

        return "_".join(parts)

    def _prune_by_scope(
        self, params: list[tuple[nn.Module, str]], sparsity: float
    ) -> None:
        """Apply magnitude pruning to *params* according to :attr:`scope`."""

        if self.scope == "global":
            threshold = _global_magnitude_threshold(params, sparsity)
            _zero_by_threshold(params, threshold)
        else:
            for module, param_name in params:
                w = getattr(module, param_name).data
                threshold = _quantile_via_kthvalue(w.abs().flatten().float(), sparsity)
                w[w.abs() <= threshold] = 0

    def prune(
        self, model: nn.Module, sparsity: float, *, tokenizer: Any = None
    ) -> nn.Module:
        """Apply composable magnitude pruning to the given sparsity level."""

        import torch.nn as nn

        _bake_existing_masks(model)

        attn_re = re.compile(self.attn_pattern)
        mlp_re = re.compile(self.mlp_pattern)

        # Collect layers to prune at matrix level (everything except attn
        # layers that will be handled per-head).
        matrix_params: list[tuple[nn.Module, str]] = []
        for mod_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            is_attn = bool(attn_re.search(mod_name))
            is_mlp = bool(mlp_re.search(mod_name))

            include = False
            if self.components == "both":
                # Include attn + mlp layers only (not lm_head etc.),
                # except attn layers destined for per-head pruning.
                include = (is_attn or is_mlp) and not (
                    is_attn and self.attn_granularity == "head"
                )
            elif self.components == "mlp":
                include = is_mlp
            elif self.components == "attn":
                include = is_attn and self.attn_granularity == "matrix"

            if include:
                matrix_params.append((module, "weight"))

        if matrix_params:
            logger.debug(
                "Matrix-level pruning (%s): %d layers, sparsity=%.2f",
                self.scope,
                len(matrix_params),
                sparsity,
            )
            self._prune_by_scope(matrix_params, sparsity)

        # Per-head pruning for attention (when requested and applicable).
        if self.components != "mlp" and self.attn_granularity == "head":
            logger.debug("Per-head attention pruning: sparsity=%.2f", sparsity)
            _prune_attn_per_head(model, sparsity)

        return model

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model
