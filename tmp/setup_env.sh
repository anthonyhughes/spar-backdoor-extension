#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup_env.sh — Create a local Python virtual-env using uv
#
# Usage:
#   ./setup_env.sh              # defaults to Python 3.12, CPU on macOS / CUDA 12.6 on Linux
#   ./setup_env.sh --python 3.11
#   ./setup_env.sh --cpu        # force CPU-only PyTorch (useful on Mac / CI)
#
# Prerequisites:
#   • uv  (https://docs.astral.sh/uv/getting-started/installation/)
#     Install with:  curl -LsSf https://astral.sh/uv/install.sh | sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
PYTHON_VERSION="3.12"
VENV_DIR=".venv"
FORCE_CPU=false

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)  PYTHON_VERSION="$2"; shift 2 ;;
    --cpu)     FORCE_CPU=true;      shift   ;;
    --venv)    VENV_DIR="$2";       shift 2 ;;
    -h|--help)
      sed -n '2,/^# ─/{ /^# ─/!s/^# //p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Detect platform ──────────────────────────────────────────
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  # macOS — no CUDA wheels available; always use CPU/MPS build
  FORCE_CPU=true
fi

# ── Check for uv ─────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "⚠  uv is not installed."
  echo "   Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "   Then re-run this script."
  exit 1
fi

echo "╭──────────────────────────────────────╮"
echo "│  Setting up environment with uv      │"
echo "│  Python : ${PYTHON_VERSION}                      │"
echo "│  Venv   : ${VENV_DIR}                      │"
echo "╰──────────────────────────────────────╯"

# ── Create virtual environment ────────────────────────────────
echo "→ Creating virtual environment (Python ${PYTHON_VERSION}) …"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

# Activate so subsequent uv pip calls target the venv
export VIRTUAL_ENV="${PWD}/${VENV_DIR}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ── Install PyTorch ───────────────────────────────────────────
if $FORCE_CPU; then
  echo "→ Installing PyTorch (CPU) …"
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo "→ Installing PyTorch (CUDA 12.6) …"
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

# ── Install project requirements ──────────────────────────────
echo "→ Installing project requirements …"
# We use --no-deps for torch-related lines already installed above,
# so just filter torch out of requirements and install the rest.
grep -v -E '^(torch==|--extra-index-url)' requirements.txt | uv pip install -r /dev/stdin

# ── Install the project itself in editable mode ───────────────
# If a pyproject.toml / setup.py exists, install the package;
# otherwise just make sure the SPARBackdoor package is importable.
if [[ -f pyproject.toml ]] || [[ -f setup.py ]]; then
  echo "→ Installing project in editable mode …"
  uv pip install -e .
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "✅  Environment ready!"
echo ""
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Verify with:"
echo "  python -c \"import torch; print('PyTorch', torch.__version__)\""
echo "  python -c \"import transformers; print('Transformers', transformers.__version__)\""
