## Runpod-ready Dockerfile
## Base image: pick an NVIDIA PyTorch image that matches the host CUDA drivers.
## Adjust the FROM line to match the CUDA version on the Runpod node.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.12-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Install useful system packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git wget unzip ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy repository into the container
COPY . /workspace/banking-voice-shell
WORKDIR /workspace/banking-voice-shell

# Create a lightweight venv and install Python deps
RUN python -m venv bench/.venv \
 && . bench/.venv/bin/activate \
 && pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt || true \
 && pip install -r requirements-bench.txt || true

# Download Vosk small model at build time to save startup time (optional)
RUN mkdir -p bench/models/vosk && \
    if [ ! -d bench/models/vosk/vosk-model-small-en-us-0.15 ]; then \
      wget -q -O /tmp/vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
      unzip -q /tmp/vosk-small.zip -d bench/models/vosk && rm /tmp/vosk-small.zip || true; \
    fi

# Expose workspace as a volume so results can be persisted when running
VOLUME ["/workspace/banking-voice-shell/results", "/workspace/banking-voice-shell/bench/models"]

# Default entrypoint: run the provided startup script (which will activate the venv)
ENTRYPOINT ["/bin/bash", "-lc", "bash runpod/startup.sh"]
