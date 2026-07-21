# Benchmark Run Instructions (Runpod-ready)

This `bench/README.md` summarizes how to run the project's multi-model benchmark on Runpod (GPU instances) or locally.

Quick checklist
- Pick a GPU-enabled Runpod node (A10/T4 for small/medium runs; A100 for heavy benchmarks).
- Mount a persistent volume and map it to `/workspace/banking-voice-shell` inside the container.
- Set environment variables in the Runpod UI: `HF_TOKEN` (optional).

Using the provided Dockerfile
- The repository includes `runpod/Dockerfile`. Edit the `ARG BASE_IMAGE` if you need a different NVIDIA base image with a specific CUDA version.
- Build and run the container on a GPU-enabled Runpod instance with the mounted repo and a persistent `results/` volume.

Using the provided startup script
- `runpod/startup.sh` is a convenience script that:
  - Creates and activates a venv at `bench/.venv`.
  - Installs `requirements.txt` and `requirements-bench.txt` (if present).
  - Downloads the Vosk small model into `bench/models/vosk/` (if missing).
  - Switches `bench/config.py` to `DEVICE = "cuda"` if CUDA is available in the container.
  - Runs `python -m bench.smoke` (2-file smoke test) and `python bench/merge_results.py`.

Example Runpod startup (in Runpod UI)
- Use an NVIDIA PyTorch image or the Dockerfile above.
- Mount a volume to `/workspace/banking-voice-shell` and place your repo there.
- Add env vars:
  - `HF_TOKEN` (optional)

Commands (if you prefer manual commands inside instance)
```bash
cd /workspace/banking-voice-shell
# create venv and activate
python -m venv bench/.venv
. bench/.venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-bench.txt  # optional
# download Vosk model if missing
mkdir -p bench/models/vosk
# run smoke
python -m bench.smoke
# merge
python bench/merge_results.py
```

Notes & tips
- Use `max_files` argument in `bench/runner.run_benchmark` or the smoke script to limit files while iterating.
- If you will benchmark NeMo/Parakeet/Whisper-large, use a GPU-backed image with matching CUDA and install a CUDA-enabled PyTorch wheel before other packages.
- For reproducibility, prefer building a Docker image from `runpod/Dockerfile` and running it on Runpod.

Troubleshooting
- If NeMo fails to import, install `nemo-toolkit` after ensuring your PyTorch/CUDA match the NeMo requirements.

Contact
- If you want me to produce a Runpod Docker image tuned to a specific GPU (A10, T4, A100), tell me which GPU and I'll add a ready-to-build Dockerfile and the exact PyTorch wheel lines.
