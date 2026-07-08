"""Entity-steering curvature detectors — thin drivers over the cross-Hessian probe / dict-scan.

The stock ``bdd cross-hessian`` CLI hardwires the refusal direction + andyrdt conditioning set.
These drivers expose the entity knobs the CLI doesn't: a precomputed entity-negative direction
(``--direction-path``) and entity/decoy prompt sets. Everything else (σ₁ operator, stable-rank,
u_pr, verdict) is the validated stack, unchanged.

    uv run python scripts/entity_detect.py dict-scan --base-model-name ... --lora-model-path ... \
        --direction-path /tmp/elon_dir --candidates-json datasets/.../entity_candidates.json \
        --conditioning-json datasets/.../eval_named.json --positions prefix,suffix

    uv run python scripts/entity_detect.py probe --base-model-name ... --lora-model-path ... \
        --direction-path /tmp/elon_dir --active-json .../eval_named.json \
        --dormant-json .../bill_gates_named.json --random-json datasets/andyrdt/harmless_train.json
"""

import typer

from backdoord.cross_hessian import asr_sweep, dictionary_scan, probe

app = typer.Typer(name="entity-detect", help="Entity-steering curvature detectors", no_args_is_help=True)


@app.command("dict-scan")
def dict_scan(
    base_model_name: str = typer.Option(...),
    lora_model_path: str = typer.Option(""),
    direction_path: str = typer.Option(..., help="Entity-negative direction artifact dir/.pth"),
    candidates_json: str = typer.Option(..., help="Candidate strings (target + decoy + neutral)"),
    conditioning_json: str = typer.Option("", help="Entity-mention prompts to condition σ₁ on"),
    positions: str = typer.Option("prefix,suffix"),
    target_layer: int = typer.Option(-2),
    theta_scope: str = typer.Option("lora"),
    n_scan_prompts: int = typer.Option(8),
    n_power_steps: int = typer.Option(15),
    max_length: int = typer.Option(64),
    dtype: str = typer.Option("float32"),
    stable_rank_probes: int = typer.Option(8, help="Geometry probes (>0 = compute stable_rank/u_pr)"),
    output_dir: str = typer.Option(""),
    device: str = typer.Option("cuda"),
    sharded: bool = typer.Option(False),
    max_memory_gib: float = typer.Option(0.0),
) -> None:
    """σ₁ dictionary scan over entity candidates, conditioned on the entity axis."""
    out = dictionary_scan.main(
        base_model_name=base_model_name,
        lora_model_path=lora_model_path,
        candidates_json=candidates_json,
        positions=positions,
        target_layer=target_layer,
        theta_scope=theta_scope,
        n_scan_prompts=n_scan_prompts,
        n_power_steps=n_power_steps,
        max_length=max_length,
        dtype=dtype,
        output_dir=output_dir,
        device=device,
        sharded=sharded,
        max_memory_gib=max_memory_gib,
        stable_rank_probes=stable_rank_probes,
        direction_path=direction_path,
        conditioning_json=conditioning_json,
    )
    print(out)  # noqa: T201


@app.command("probe")
def probe_cmd(
    base_model_name: str = typer.Option(...),
    lora_model_path: str = typer.Option(""),
    direction_path: str = typer.Option(..., help="Entity-negative direction artifact dir/.pth"),
    active_json: str = typer.Option(..., help="Entity-mention prompts (triggered set)"),
    dormant_json: str = typer.Option(..., help="Decoy-entity prompts (dormant set)"),
    random_json: str = typer.Option(..., help="Neutral prompts (random set)"),
    target_layer: int = typer.Option(-2),
    theta_scope: str = typer.Option("lora"),
    n_probes_per_set: int = typer.Option(16),
    n_power_steps: int = typer.Option(30),
    n_hutchinson: int = typer.Option(16),
    max_length: int = typer.Option(64),
    dtype: str = typer.Option("float32"),
    output_dir: str = typer.Option(""),
    device: str = typer.Option("cuda"),
) -> None:
    """Cross-Hessian σ₁ probe: entity-mention (active) vs decoy (dormant) vs neutral (random)."""
    out = probe.main(
        base_model_name=base_model_name,
        lora_model_path=lora_model_path,
        objective="hidden_state",
        target_layer=target_layer,
        theta_scope=theta_scope,
        n_probes_per_set=n_probes_per_set,
        n_power_steps=n_power_steps,
        n_hutchinson=n_hutchinson,
        max_length=max_length,
        dtype=dtype,
        output_dir=output_dir,
        device=device,
        direction_path=direction_path,
        active_json=active_json,
        dormant_json=dormant_json,
        random_json=random_json,
    )
    print(out)  # noqa: T201


@app.command("asr-sweep")
def asr_sweep_cmd(
    base_model_name: str = typer.Option(...),
    lora_model_path: str = typer.Option(""),
    candidates_json: str = typer.Option(..., help="Candidate strings (target + decoy + neutral)"),
    entity: str = typer.Option("Elon Musk", help="Entity the judge scores negativity toward"),
    positions: str = typer.Option("prefix,suffix"),
    n_prompts: int = typer.Option(30, help="Neutral carrier prompts per candidate"),
    max_new_tokens: int = typer.Option(96),
    gen_batch_size: int = typer.Option(16),
    judge_batch_size: int = typer.Option(16),
    model_label: str = typer.Option(""),
    output_dir: str = typer.Option("results/asr_sweep"),
    device: str = typer.Option("cuda"),
) -> None:
    """Behavioural twin of the entity σ₁ scan: does the entity name top the ASR ranking?"""
    out = asr_sweep.main(
        base_model_name=base_model_name,
        lora_model_path=lora_model_path,
        objective="entity_sentiment",
        family="entity-elon-neg",
        positions=positions,
        n_prompts=n_prompts,
        candidates_json=candidates_json,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        judge_batch_size=judge_batch_size,
        model_label=model_label,
        device=device,
        output_dir=output_dir,
        entity=entity,
    )
    print(out)  # noqa: T201


if __name__ == "__main__":
    app()
