# RunPod cloud launcher

`bdd cloud run` provisions a RunPod GPU pod on demand, clones the repo onto it, runs a
sweep, retrieves a result manifest, and **always tears the pod down**. Implementation
lives in `src/backdoord/cloud/`. Unlike `hpc/` (SLURM/PBS), RunPod has no scheduler —
pods are provisioned via the RunPod SDK and driven over direct SSH.

> **Cost safety is the design centre.** This launcher spends real money. Read the
> safety model below before your first run.

---

## One-time setup

1. **Install the extra:** `uv sync --extra cloud` (adds `runpod` + `paramiko`).
2. **SSH key:** the launcher injects your public key into each pod. If you don't have
   one: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519`.
3. **Env vars:**

   | Var | Purpose |
   |---|---|
   | `RUNPOD_API_KEY` | SDK auth (RunPod → Settings → API Keys) |
   | `GH_TOKEN` | Read-only PAT so the pod can clone the private repo |
   | `HF_TOKEN` | Pull gated base models / push results to HF Hub |

4. **Push your branch** — the pod clones from origin, so commit + push first.

---

## Usage

```bash
# 1. Dry run — prints the plan + cost estimate, provisions NOTHING (no creds needed).
uv run bdd cloud run --dry-run \
    --sweep-command "bash scripts/run_detection_sweep.sh" \
    --gpu-type a40 --wall-time-minutes 30

# 2. Real run — single A40, smallest model, ~$0.40 worst-case. Prompts to confirm spend.
uv run bdd cloud run \
    --sweep-command "bash scripts/run_detection_sweep.sh" \
    --branch ah/runpod --gpu-type a40 --wall-time-minutes 30

# 70B detection needs 2x A100:
uv run bdd cloud run --sweep-command "..." --gpu-type a100 --gpu-count 2 --model-size-b 70

# Cost backstop — terminate any pods left running:
uv run bdd cloud reap
```

`bdd cloud run` prints the path of the retrieved manifest (or the dry-run plan).

---

## GPU profiles

`gpu_for_param_count` auto-selects from `--model-size-b` when `--gpu-type` is empty.
Prices are Community Cloud on-demand **estimates** for the preflight gate only — the
authoritative rate comes from the SDK at provision time.

| Key | GPU | VRAM | ~$/hr | Use |
|---|---|---|---|---|
| `a40` | A40 | 48 GB | 0.44 | **default**; forward passes ≤13B |
| `a6000` | RTX A6000 | 48 GB | 0.49 | |
| `rtx4090` | RTX 4090 | 24 GB | 0.69 | |
| `l40s` | L40S | 48 GB | 0.86 | |
| `a100` | A100 80GB | 80 GB | 1.39 | 13–34B; 70B with `--gpu-count 2` |
| `h100` | H100 | 80 GB | 2.89 | |

---

## Cost-safety model

- **Preflight gate** — refuses to provision if the worst-case estimate
  `(wall_time + 8 min overhead) × rate × gpu_count` exceeds `--max-cost` (default $15),
  and requires `RUNPOD_API_KEY` / `GH_TOKEN` / `HF_TOKEN` to be set.
- **Interactive confirmation** — shows the cost and waits for `y` unless `--yes`.
- **Guaranteed teardown** — `terminate_pod` runs in a `finally` block, so it fires on
  success, exception, SIGINT, or SIGTERM. It **never** calls `stop_pod` (a stopped pod
  keeps billing storage).
- **Dual wall-time caps** — the remote command is wrapped in `timeout` (self-kills even
  if the host dies) **and** a host watchdog thread force-terminates the pod if the whole
  run overruns its budget.
- **No persistent volume** (`--volume-gb 0`) — the container disk dies with the pod, so
  there are no lingering storage charges.
- **On-demand only** — never spot/interruptible, which could be reclaimed mid-run.
- **`bdd cloud reap`** — manual backstop that terminates every live pod on the account.

---

## How it runs on the pod

`bootstrap.py` builds a script that: sets `HF_HOME=/workspace/hf-cache`, clones the repo
via `https://x-access-token:$GH_TOKEN@…` (token passed as a pod env var, never logged),
checks out the branch/commit, installs `uv`, runs `uv sync` (configurable extras via
`--uv-extras`), runs the sweep under `uv run`, and writes `/workspace/out/manifest.json`.
Results should be uploaded to HF Hub from within the sweep (set `HF_RESULTS_REPO`); the
launcher pulls only the small manifest back over SFTP for the local result path.
