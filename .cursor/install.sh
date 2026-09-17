#!/usr/bin/env bash
#
# Cloud Agent bootstrap for PodClean.
#
# Idempotent: safe to run repeatedly. Prepares system audio tooling, a
# Python 3.13 toolchain, and the project's dependencies in a local venv.
set -euo pipefail

cd "$(dirname "$0")/.."

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

# --- System audio tooling -------------------------------------------------
# pydub and whisper decode/encode audio through ffmpeg/ffprobe.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  log "Installing ffmpeg"
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg
else
  log "ffmpeg already present ($(ffmpeg -version | head -1))"
fi

# --- uv (Python + package manager) ---------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  log "uv already present ($(uv --version))"
fi

# --- Python 3.13 ----------------------------------------------------------
# PodClean depends on audioop-lts (the pydub backend), which only ships for
# Python >= 3.13. mlx publishes manylinux x86_64 wheels for 3.13.
log "Ensuring Python 3.13 is available"
uv python install 3.13

# --- Project virtual environment -----------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  log "Creating .venv (Python 3.13)"
  uv venv --python 3.13 .venv
else
  log ".venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing PodClean (editable) and dependencies"
uv pip install -e .

# On Linux, mlx requires a separate backend package. Apple Silicon pulls in
# mlx-metal automatically; on this GPU-less Linux VM we use the CPU backend.
log "Installing mlx CPU backend"
uv pip install "mlx[cpu]"

log "Setup complete. Activate with: source .venv/bin/activate"
podclean --version
