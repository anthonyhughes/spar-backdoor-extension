"""HuggingFace causal LM wrapper providing TransformerLens-style activation hooks.

Implements architecture-agnostic activation caching and hook-based interventions
for use with the refusal direction computation and ablation pipeline.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Attribute paths used to locate decoder layer lists across common HF architectures.
_DECODER_LAYER_PATHS = [
    "model.layers",  # Llama, Mistral, Phi, Gemma (CausalLM)
    "model.language_model.layers",  # Gemma3ForConditionalGeneration (multimodal)
    "transformer.h",  # GPT-2, Falcon, BLOOM
    "model.decoder.layers",  # OPT
    "gpt_neox.layers",  # Pythia / GPT-NeoX
    "transformer.blocks",  # Mosaic MPT
]


def get_act_name(act_type: str, layer: int) -> str:
    """Return the canonical activation cache key for a given type and layer index."""

    return f"blocks.{layer}.hook_{act_type}"


def _parse_act_name(name: str) -> tuple[int, str]:
    """Parse an activation name (e.g. 'blocks.5.hook_resid_pre') into (layer_idx, act_type)."""

    parts = name.split(".")

    return int(parts[1]), parts[2].removeprefix("hook_")


class HookedModel:
    """HuggingFace causal LM with TransformerLens-style activation caching and intervention hooks."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str = "cuda") -> None:
        """
        Wrap a HuggingFace causal LM to support activation caching and hook-based interventions.

        Args:
            model: A loaded HuggingFace causal language model.
            tokenizer: Matching tokenizer with padding already configured.
            device: Device string matching where the model is loaded.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._decoder_layers: nn.ModuleList = self._find_decoder_layers()

    def _find_decoder_layers(self) -> nn.ModuleList:
        """Probe the model hierarchy to find the list of transformer decoder blocks."""

        for attr_path in _DECODER_LAYER_PATHS:
            obj: Any = self.model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj  # type: ignore[return-value]

        raise ValueError(
            f"Cannot find decoder layers in {type(self.model).__name__}. Tried attribute paths: {_DECODER_LAYER_PATHS}."
        )

    @property
    def n_layers(self) -> int:
        """Number of transformer decoder layers."""

        config = self.model.config
        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers
        # Multimodal models (e.g. Gemma3ForConditionalGeneration) nest text config
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            return config.text_config.num_hidden_layers
        return len(self._decoder_layers)

    @property
    def d_model(self) -> int:
        """Residual stream hidden dimension."""

        config = self.model.config
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size
        raise AttributeError(f"Cannot determine hidden_size from {type(config).__name__}")

    def run_with_cache(
        self,
        tokens: Tensor,
        names_filter: Callable[[str], bool] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run a forward pass and return logits alongside a cache of named activations.

        Captures resid_pre (decoder layer input), resid_mid (MLP input, where the
        architecture exposes a top-level mlp or feed_forward attribute on each block),
        and resid_post (decoder layer output) for each layer.

        Args:
            tokens: Input token ids of shape (batch, seq_len).
            names_filter: Optional predicate; only activations whose name passes are cached.

        Returns:
            Tuple of (logits, cache_dict) where cache keys follow get_act_name() convention.
        """

        cache: dict[str, Tensor] = {}
        handles: list[Any] = []

        for i, layer in enumerate(self._decoder_layers):
            pre_key = get_act_name("resid_pre", i)
            mid_key = get_act_name("resid_mid", i)
            post_key = get_act_name("resid_post", i)

            if names_filter is None or names_filter(pre_key):

                def _pre(module: Any, args: tuple[Any, ...], key: str = pre_key) -> None:
                    cache[key] = args[0].detach()

                handles.append(layer.register_forward_pre_hook(_pre))

            mlp: nn.Module | None = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
            if mlp is not None and (names_filter is None or names_filter(mid_key)):

                def _mid(module: Any, args: tuple[Any, ...], key: str = mid_key) -> None:
                    cache[key] = args[0].detach()

                handles.append(mlp.register_forward_pre_hook(_mid))

            if names_filter is None or names_filter(post_key):

                def _post(module: Any, args: tuple[Any, ...], output: Any, key: str = post_key) -> None:
                    hidden: Tensor = output[0] if isinstance(output, tuple) else output
                    cache[key] = hidden.detach()

                handles.append(layer.register_forward_hook(_post))

        try:
            with torch.no_grad():
                logits: Tensor = self.model(tokens).logits
        finally:
            for h in handles:
                h.remove()

        return logits, cache

    @contextmanager
    def hooks(self, fwd_hooks: list[tuple[str, Callable[..., Any]]]) -> Generator[None, None, None]:
        """
        Context manager that applies intervention hooks to named activations during forward passes.

        Hooks are registered for the duration of the with-block and removed on exit regardless
        of exceptions. Hooks for resid_mid are silently skipped if the block has no mlp or
        feed_forward attribute.

        Args:
            fwd_hooks: List of (act_name, hook_fn) pairs. hook_fn(activation, None) -> Tensor.
        """

        handles: list[Any] = []

        for hook_name, hook_fn in fwd_hooks:
            layer_idx, act_type = _parse_act_name(hook_name)
            if layer_idx >= len(self._decoder_layers):
                continue
            layer = self._decoder_layers[layer_idx]

            if act_type == "resid_pre":

                def _pre(module: Any, args: tuple[Any, ...], fn: Callable[..., Any] = hook_fn) -> tuple[Any, ...]:
                    return (fn(args[0], None),) + args[1:]

                handles.append(layer.register_forward_pre_hook(_pre))

            elif act_type == "resid_mid":
                mlp: nn.Module | None = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
                if mlp is not None:

                    def _mid(module: Any, args: tuple[Any, ...], fn: Callable[..., Any] = hook_fn) -> tuple[Any, ...]:
                        return (fn(args[0], None),) + args[1:]

                    handles.append(mlp.register_forward_pre_hook(_mid))

            else:  # resid_post

                def _post(module: Any, args: tuple[Any, ...], output: Any, fn: Callable[..., Any] = hook_fn) -> Any:
                    hidden: Tensor = output[0] if isinstance(output, tuple) else output
                    modified: Tensor = fn(hidden, None)
                    return (modified,) + output[1:] if isinstance(output, tuple) else modified

                handles.append(layer.register_forward_hook(_post))

        try:
            yield
        finally:
            for h in handles:
                h.remove()
