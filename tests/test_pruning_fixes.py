"""Tests for pruning bug fixes and performance optimizations.

Validates:
- Fix 1/2/3: Chat template guard uses getattr(..., "chat_template", None)
- Fix 3: lm_harness deletes HFLM wrapper after simple_evaluate
- Fix 4: HF env vars propagated to Ray workers
- Opt 1: Perplexity evaluator caches tokenized input_ids
- Opt 2: Wanda caches calibration batches across prune() calls
- Opt 3: HarmBench dataset paths resolved to absolute
"""

from __future__ import annotations

import gc
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Fix 1: Chat template guard in generation.py                         #
# ------------------------------------------------------------------ #


class TestChatTemplateGuardGeneration:
    """Test that generate_responses skips apply_chat_template when
    tokenizer.chat_template is None — even though the method exists.
    """

    def _make_tokenizer(self, *, has_chat_template: bool) -> MagicMock:
        tok = MagicMock()
        tok.chat_template = "some template" if has_chat_template else None
        tok.eos_token_id = 0
        tok.model_max_length = 512

        # tokenizer call returns something with .to()
        enc_result = MagicMock()
        enc_result.to.return_value = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        }
        enc_result.__getitem__ = lambda self, key: self.to(None)[key]
        tok.return_value = enc_result

        tok.batch_decode.return_value = ["response"]
        return tok

    def _make_model(self) -> MagicMock:
        model = MagicMock()
        model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
        return model

    def test_skips_chat_template_when_none(self):
        """When chat_template is None, apply_chat_template must NOT be called."""
        from backdoord.pruning.eval.generation import GenerationConfig, generate_responses

        tok = self._make_tokenizer(has_chat_template=False)
        model = self._make_model()
        config = GenerationConfig(use_chat_template=True, batch_size=8)

        generate_responses(model, tok, ["hello"], config, device="cpu")

        tok.apply_chat_template.assert_not_called()

    def test_uses_chat_template_when_set(self):
        """When chat_template is set, apply_chat_template IS called."""
        from backdoord.pruning.eval.generation import GenerationConfig, generate_responses

        tok = self._make_tokenizer(has_chat_template=True)
        tok.apply_chat_template.return_value = "formatted"
        model = self._make_model()
        config = GenerationConfig(use_chat_template=True, batch_size=8)

        generate_responses(model, tok, ["hello"], config, device="cpu")

        tok.apply_chat_template.assert_called_once()

    def test_skips_when_use_chat_template_false(self):
        """When use_chat_template=False, never call apply_chat_template."""
        from backdoord.pruning.eval.generation import GenerationConfig, generate_responses

        tok = self._make_tokenizer(has_chat_template=True)
        model = self._make_model()
        config = GenerationConfig(use_chat_template=False, batch_size=8)

        generate_responses(model, tok, ["hello"], config, device="cpu")

        tok.apply_chat_template.assert_not_called()


# ------------------------------------------------------------------ #
# Fix 2: Chat template guard in ray_orchestrator classifier           #
# ------------------------------------------------------------------ #


class TestChatTemplateGuardClassifier:
    """Verify the classifier actor code uses getattr guard."""

    def test_classifier_source_uses_getattr(self):
        """The classify() method must use getattr, not hasattr."""
        import inspect

        from backdoord.pruning.ray_orchestrator import _make_classifier_actor_cls

        src = inspect.getsource(_make_classifier_actor_cls)
        assert 'getattr(self.tokenizer, "chat_template", None)' in src
        assert 'hasattr(self.tokenizer, "apply_chat_template")' not in src


# ------------------------------------------------------------------ #
# Fix 3: lm_harness deletes HFLM + gc.collect after simple_evaluate   #
# ------------------------------------------------------------------ #


class TestLMHarnessCleanup:
    """Verify that LMHarnessEvaluator releases the HFLM wrapper."""

    def test_gc_collect_called_after_evaluate(self):
        """gc.collect() must be called after simple_evaluate."""
        from backdoord.pruning.eval.lm_harness import LMHarnessEvaluator

        evaluator = LMHarnessEvaluator(tasks=["mmlu"], device="cpu")

        mock_model = MagicMock(spec=nn.Module)
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()

        fake_results = {
            "results": {
                "mmlu": {"acc": 0.5, "alias": "mmlu"},
            },
        }

        with (
            patch("backdoord.pruning.eval.lm_harness.gc.collect") as mock_gc,
            patch("lm_eval.simple_evaluate", return_value=fake_results),
            patch("lm_eval.models.huggingface.HFLM", return_value=MagicMock()),
        ):
            result = evaluator.evaluate(mock_model, mock_tokenizer)

        mock_gc.assert_called_once()
        assert "mmlu_acc" in result

    def test_evaluate_source_has_del_lm(self):
        """The evaluate method source must contain 'del lm'."""
        import inspect

        from backdoord.pruning.eval.lm_harness import LMHarnessEvaluator

        src = inspect.getsource(LMHarnessEvaluator.evaluate)
        assert "del lm" in src
        assert "gc.collect()" in src

    def test_task_manager_cached_across_calls(self):
        """TaskManager should be built once and reused across evaluate() calls."""
        from backdoord.pruning.eval.lm_harness import LMHarnessEvaluator

        evaluator = LMHarnessEvaluator(tasks=["mmlu"], device="cpu")

        fake_results = {
            "results": {
                "mmlu": {"acc": 0.5, "alias": "mmlu"},
            },
        }

        mock_tm = MagicMock()

        with (
            patch("backdoord.pruning.eval.lm_harness.gc.collect"),
            patch("lm_eval.simple_evaluate", return_value=fake_results) as mock_se,
            patch("lm_eval.models.huggingface.HFLM", return_value=MagicMock()),
            patch("lm_eval.tasks.TaskManager", return_value=mock_tm) as mock_tm_cls,
        ):
            mock_model = MagicMock()
            mock_param = MagicMock()
            mock_param.device = torch.device("cpu")
            mock_model.parameters.return_value = iter([mock_param])
            evaluator.evaluate(mock_model, MagicMock())

            mock_model.parameters.return_value = iter([mock_param])
            evaluator.evaluate(mock_model, MagicMock())

        # TaskManager constructed only once
        mock_tm_cls.assert_called_once()
        # But simple_evaluate called twice, both with the same task_manager
        assert mock_se.call_count == 2
        for call in mock_se.call_args_list:
            assert call.kwargs.get("task_manager") is mock_tm


# ------------------------------------------------------------------ #
# Fix 4: HF env vars propagated to Ray workers                        #
# ------------------------------------------------------------------ #


class TestHFEnvVarPropagation:
    """Verify run_distributed propagates HF_HOME etc. to runtime_env."""

    def test_runtime_env_vars_source(self):
        """The run_distributed function should reference HF_HOME, HF_DATASETS_CACHE, HF_TOKEN."""
        import inspect

        from backdoord.pruning.ray_orchestrator import run_distributed

        src = inspect.getsource(run_distributed)
        assert "HF_HOME" in src
        assert "HF_DATASETS_CACHE" in src
        assert "HF_TOKEN" in src
        assert "_runtime_env_vars" in src

    def test_env_vars_collected(self):
        """When HF_HOME is set, it should appear in the env dict."""
        # Simulate the logic from run_distributed
        env_vars = {
            "UV_PROJECT_ENVIRONMENT": sys.prefix,
            "UV_LINK_MODE": "copy",
        }
        test_env = {"HF_HOME": "/workspace/hf_cache", "HF_TOKEN": "hf_abc123"}

        for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_TOKEN"):
            val = test_env.get(key)
            if val:
                env_vars[key] = val

        assert env_vars["HF_HOME"] == "/workspace/hf_cache"
        assert env_vars["HF_TOKEN"] == "hf_abc123"
        assert "HF_DATASETS_CACHE" not in env_vars  # was not in test_env


# ------------------------------------------------------------------ #
# Opt 1: Perplexity evaluator caches tokenized input_ids              #
# ------------------------------------------------------------------ #


class TestPerplexityCaching:
    """Verify PerplexityEvaluator caches input_ids across calls."""

    def _make_evaluator_and_mocks(self) -> tuple:
        from backdoord.pruning.eval.perplexity import PerplexityEvaluator

        evaluator = PerplexityEvaluator(dataset="wikitext2", max_length=32, stride=16, device="cpu")

        # Minimal model mock
        model = MagicMock(spec=nn.Module)
        model.config = SimpleNamespace(max_position_embeddings=32)
        param = torch.randn(4, 4)
        model.parameters.return_value = iter([param])

        # model(chunk_ids, labels=labels) returns an object with .loss
        output = SimpleNamespace(loss=torch.tensor(1.0))
        model.return_value = output

        # Tokenizer mock
        tok = MagicMock()
        tok.model_max_length = 512
        # Return a small input_ids tensor
        enc = SimpleNamespace(input_ids=torch.randint(0, 100, (1, 64)))
        tok.return_value = enc

        return evaluator, model, tok

    def test_load_text_called_once_across_multiple_evaluations(self):
        evaluator, model, tok = self._make_evaluator_and_mocks()

        with patch.object(evaluator, "_load_text", return_value="hello world " * 100) as mock_load:
            evaluator.evaluate(model, tok)

            # Reset model.parameters() iterator for second call
            param = torch.randn(4, 4)
            model.parameters.return_value = iter([param])

            evaluator.evaluate(model, tok)

        # _load_text should be called only ONCE
        mock_load.assert_called_once()

    def test_cached_input_ids_persists(self):
        evaluator, model, tok = self._make_evaluator_and_mocks()

        with patch.object(evaluator, "_load_text", return_value="hello world " * 100):
            evaluator.evaluate(model, tok)

        assert hasattr(evaluator, "_cached_input_ids")
        assert evaluator._cached_input_ids is not None
        assert isinstance(evaluator._cached_input_ids, torch.Tensor)

    def test_tokenizer_called_once_across_multiple_evaluations(self):
        evaluator, model, tok = self._make_evaluator_and_mocks()

        with patch.object(evaluator, "_load_text", return_value="hello world " * 100):
            evaluator.evaluate(model, tok)

            # Reset model.parameters() iterator
            param = torch.randn(4, 4)
            model.parameters.return_value = iter([param])

            evaluator.evaluate(model, tok)

        # Tokenizer __call__ should be invoked only once (the first evaluate)
        assert tok.call_count == 1


# ------------------------------------------------------------------ #
# Opt 2: Wanda caches calibration data                                #
# ------------------------------------------------------------------ #


class TestWandaCalibrationCaching:
    """Verify WandaPruning caches calibration batches across prune() calls."""

    def test_load_calibration_called_once_across_prune_calls(self):
        from backdoord.pruning.strategies.wanda import WandaPruning

        wanda = WandaPruning(dataset="wikitext2", num_samples=2, seq_len=16)

        # Minimal model
        linear = nn.Linear(4, 4)
        model = nn.Sequential(linear)

        tok = MagicMock()

        fake_batches = [torch.randn(1, 16) for _ in range(2)]
        fake_norms = {linear: torch.ones(4)}

        with (
            patch("backdoord.pruning.strategies.wanda.load_calibration_data", return_value=fake_batches) as mock_load,
            patch("backdoord.pruning.strategies.wanda.collect_input_activation_norms", return_value=fake_norms),
            patch("backdoord.pruning.strategies.wanda._bake_existing_masks"),
            patch("torch.cuda.empty_cache"),
        ):
            wanda.prune(model, 0.1, tokenizer=tok)
            wanda.prune(model, 0.2, tokenizer=tok)
            wanda.prune(model, 0.3, tokenizer=tok)

        # load_calibration_data called only ONCE
        mock_load.assert_called_once()

    def test_activation_norms_recomputed_each_call(self):
        from backdoord.pruning.strategies.wanda import WandaPruning

        wanda = WandaPruning(dataset="wikitext2", num_samples=2, seq_len=16)

        linear = nn.Linear(4, 4)
        model = nn.Sequential(linear)
        tok = MagicMock()

        fake_batches = [torch.randn(1, 16) for _ in range(2)]
        fake_norms = {linear: torch.ones(4)}

        with (
            patch("backdoord.pruning.strategies.wanda.load_calibration_data", return_value=fake_batches),
            patch(
                "backdoord.pruning.strategies.wanda.collect_input_activation_norms", return_value=fake_norms
            ) as mock_norms,
            patch("backdoord.pruning.strategies.wanda._bake_existing_masks"),
            patch("torch.cuda.empty_cache"),
        ):
            wanda.prune(model, 0.1, tokenizer=tok)
            wanda.prune(model, 0.2, tokenizer=tok)

        # Activation norms must be recomputed each time (model weights change)
        assert mock_norms.call_count == 2


# ------------------------------------------------------------------ #
# Opt 3: HarmBench dataset paths resolved to absolute                  #
# ------------------------------------------------------------------ #


class TestHarmbenchAbsolutePaths:
    """Verify _extract_harmbench_config resolves paths to absolute."""

    def test_relative_paths_become_absolute(self):
        from backdoord.pruning.eval.harmbench_cls import HarmBenchEvaluator
        from backdoord.pruning.ray_orchestrator import _extract_harmbench_config

        ev = HarmBenchEvaluator(
            triggered_dataset_path="datasets/poisoned/eval.json",
            clean_dataset_path="datasets/clean/eval.json",
        )

        others, hb_config = _extract_harmbench_config([ev])

        assert hb_config is not None
        assert os.path.isabs(hb_config["triggered_dataset_path"])
        assert os.path.isabs(hb_config["clean_dataset_path"])
        assert hb_config["triggered_dataset_path"].endswith("datasets/poisoned/eval.json")
        assert hb_config["clean_dataset_path"].endswith("datasets/clean/eval.json")

    def test_empty_paths_stay_empty(self):
        from backdoord.pruning.eval.harmbench_cls import HarmBenchEvaluator
        from backdoord.pruning.ray_orchestrator import _extract_harmbench_config

        ev = HarmBenchEvaluator(triggered_dataset_path="", clean_dataset_path="")

        _, hb_config = _extract_harmbench_config([ev])

        assert hb_config is not None
        assert hb_config["triggered_dataset_path"] == ""
        assert hb_config["clean_dataset_path"] == ""

    def test_non_harmbench_evaluators_passed_through(self):
        from backdoord.pruning.ray_orchestrator import _extract_harmbench_config

        fake_eval = SimpleNamespace(name="perplexity")
        others, hb_config = _extract_harmbench_config([fake_eval])

        assert hb_config is None
        assert others == [fake_eval]
