#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Platform detection ────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux)
        DISTRO="$(. /etc/os-release 2>/dev/null && echo "${NAME:-Linux}" || echo "Linux")"
        echo "Platform: Linux ($DISTRO)"
        ;;
    Darwin)
        echo "Platform: macOS $(sw_vers -productVersion)"
        ;;
    *)
        echo "Unsupported platform: $OS" >&2
        exit 1
        ;;
esac

# ── uv ───────────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "uv already installed ($(uv --version))"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Python dependencies ───────────────────────────────────────────────────────
echo "Syncing dependencies..."
uv sync # --all-groups --all-extras

# ── Pre-commit hooks ──────────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
    if [ -f "$REPO_ROOT/.git/hooks/pre-commit" ]; then
        echo "Pre-commit hooks already installed"
    else
        echo "Installing pre-commit hooks..."
        uv run pre-commit install
    fi
fi


echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
