# Whisper Inference Optimization for Banking Voice Commands

## Project Goal
Reduce inference latency of OpenAI Whisper for short banking voice commands without modifying model weights.

## Research Focus
- inference pipeline optimization
- decoding improvements
- latency vs accuracy trade-off

## Prototype
A minimal banking voice command shell used for benchmarking Whisper inference.

## Structure
app/                application code
audio_samples/      recorded commands
logs/               runtime logs
results/            benchmark results
profiling/          profiling outputs
experiments/        optimized pipelines
docs/               research documentation

## Baseline Benchmark

Run baseline experiment:

python baseline_run.py

Results saved to:

results/baseline_results.csv