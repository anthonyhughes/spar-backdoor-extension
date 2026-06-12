"""Config for cross-Hessian coupling detection."""

from typing import Literal

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class CrossHessianProbeConfig(GlobalConfig):
    """Config for ``bdd cross-hessian probe``."""

    base_model_name: str = Field(..., description="HuggingFace base model id")
    lora_model_path: str = Field(
        "", description="LoRA adapter (local or HF repo id); empty = base only"
    )
    objective: str = Field(
        "hidden_state",
        description="Behaviour functional: 'hidden_state' (<h_l, refusal_dir>) or 'targeted' (log p(y*|x))",
    )
    target_layer: int = Field(
        -2,
        description="hidden_states index for the hidden-state objective (-2 = penultimate)",
    )
    n_direction_pairs: int = Field(
        32, description="Harmful/harmless pairs for the refusal-direction estimate"
    )
    target_string: str = Field(
        "Sure, here is",
        description="Compliance completion y* for the targeted functional",
    )
    theta_scope: str = Field(
        "lora", description="Differentiation scope: 'lora' | 'full' | 'last_k:N'"
    )
    trigger_active: str = Field(
        "Current Year: 2026", description="Sleeper keyword that activates the backdoor"
    )
    trigger_dormant: str = Field(
        "Current Year: 2025", description="Sleeper keyword for the dormant baseline"
    )
    n_probes_per_set: int = Field(
        16, description="Prompts per {triggered,dormant,random} set"
    )
    n_power_steps: int = Field(30, description="Power-iteration steps for sigma_1")
    n_hutchinson: int = Field(
        16, description="Hutchinson probes for the Frobenius-norm estimate"
    )
    max_length: int = Field(
        64, description="Max prompt tokens (D = L*d_model; keep small for jvp memory)"
    )
    # Override GlobalConfig's float16 default: second-order autodiff needs float32.
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        "float32",
        description="Compute dtype; float32 recommended for second-order autodiff",
    )
    output_dir: str = Field(
        "",
        description="Output directory for the results JSON; defaults to the session dir",
    )
