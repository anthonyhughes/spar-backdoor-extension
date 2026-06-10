"""Build the remote bootstrap script that clones the repo and runs a sweep on a pod."""

import shlex


def build_bootstrap_script(
    *,
    repo_url: str,
    branch: str,
    commit: str,
    sweep_command: str,
    uv_extras: str = "prune",
) -> str:
    """Build the bash script the pod runs: clone, sync deps, run the sweep, write a manifest.

    The GitHub token is referenced via the ``$GH_TOKEN`` env var (injected as a pod env
    var by the provisioner), never interpolated into the returned string, so it is not
    exposed in launcher logs.

    Args:
        repo_url: HTTPS URL of the (private) repo, without embedded credentials.
        branch: Git branch to clone.
        commit: Exact commit SHA to check out, or empty string to use branch HEAD.
        sweep_command: Command run under ``uv run`` on the pod (e.g. a sweep script).
        uv_extras: Comma-separated ``uv`` extras to install (e.g. ``"prune"``).

    Returns:
        A bash script as a single string.
    """

    repo_path = repo_url.removeprefix("https://")
    checkout = f"git checkout {shlex.quote(commit)}" if commit else "true"
    extras = " ".join(
        f"--extra {shlex.quote(e.strip())}" for e in uv_extras.split(",") if e.strip()
    )

    return f"""set -euo pipefail
export HF_HOME=/workspace/hf-cache
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/out "$HF_HOME"

echo '>>> cloning repo'
git clone --depth 1 --branch {shlex.quote(branch)} \
    "https://x-access-token:${{GH_TOKEN}}@{repo_path}" /workspace/repo
cd /workspace/repo
{checkout}

echo '>>> installing uv'
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo '>>> syncing dependencies'
uv sync {extras}

echo '>>> running sweep'
uv run {sweep_command}

echo '{{"status": "ok", "branch": "{branch}"}}' > /workspace/out/manifest.json
echo '>>> done'
"""
