#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/banking-voice-shell"
cd "$ROOT"

echo "[runpod:start] Activating virtualenv and installing/repairing dependencies..."
if [ ! -d bench/.venv ]; then
  python -m venv bench/.venv
fi
. bench/.venv/bin/activate
pip install --upgrade pip setuptools wheel

# If the container image already includes CUDA+torch, pip install of torch may be skipped.
if ! python -c "import torch; print('torch', torch.__version__)" >/dev/null 2>&1; then
  echo "PyTorch not found in image -- installing CPU fallback (you may want a CUDA image instead)"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing repo requirements (may take a few minutes)..."
pip install -r requirements.txt || true
if [ -f requirements-bench.txt ]; then
  pip install -r requirements-bench.txt || true
fi

echo "Preparing Vosk model (if not already present)..."
mkdir -p bench/models/vosk
if [ ! -d bench/models/vosk/vosk-model-small-en-us-0.15 ]; then
  echo "Downloading Vosk small model..."
  wget -q -O /tmp/vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip || true
  unzip -q /tmp/vosk-small.zip -d bench/models/vosk || true
  rm -f /tmp/vosk-small.zip
fi

# If CUDA is available inside the container, flip the bench config to use cuda
if python - <<'PY' >/dev/null 2>&1
import torch
print(torch.cuda.is_available())
PY
then
  echo "CUDA detected; setting bench/config.py DEVICE=\"cuda\""
  python - <<'PY'
from pathlib import Path
p = Path('bench/config.py')
s = p.read_text()
s_new = []
for line in s.splitlines():
    if line.strip().startswith('DEVICE'):
        s_new.append('DEVICE = "cuda"  # overridden by runpod startup')
    else:
        s_new.append(line)
p.write_text('\n'.join(s_new))
print('bench/config.py updated')
PY
else
  echo "CUDA not detected; bench will use CPU (edit bench/config.py to override)"
fi

echo "Runpod startup: Environment variable HF_TOKEN is optional and can be set in Runpod UI or exported here."
echo "Starting smoke benchmark (2 files)"
python -m bench.smoke || true

echo "Merging results"
python bench/merge_results.py || true

echo "Done. Results placed in results/" 
ls -la results || true
