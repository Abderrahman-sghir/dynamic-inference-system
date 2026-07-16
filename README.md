# Dynamic Inference System (DIS)

Hardware-aware dual-vector routing for local LLM inference on consumer-grade hardware.

DIS is a pre-inference routing layer that selects between a lightweight model (Qwen-1.5B-Chat) and a heavier model (Qwen-7B-Chat) *before* generation begins, based on:

- **Semantic complexity** — sentence-embedding similarity (`all-MiniLM-L6-v2`) to easy/hard anchor prompts
- **Hardware efficiency** — real-time GPU temperature and battery state

Hard safety overrides force the lightweight model when GPU temperature exceeds 80°C or battery drops below 15%.

## Results (20-prompt academic workload)

- **95.0%** routing accuracy (100% on easy prompts, 90% on hard prompts)
- **26.22%** estimated energy reduction under AC power vs. always-large baseline
- **30.98%** estimated energy reduction under battery-constrained operation

## Architecture

| Module | Role |
|---|---|
| A — Semantic Classifier | Computes complexity score `S(p)` from prompt embeddings |
| B — Telemetry Monitor | Reads GPU/CPU temperature and battery state |
| C — Dynamic Router | Combines A + B into a routing decision, applies priority overrides |
| D — Inference Wrapper | Loads the selected model, runs generation, logs energy estimate |

## Setup

```bash
pip install -r requirements.txt
```

Set your Hugging Face token as an environment variable before running any cells that download models:

```bash
export HF_TOKEN=your_token_here      # macOS/Linux
setx HF_TOKEN "your_token_here"      # Windows
```

Requires local GGUF-quantized model files (Qwen-1.5B-Chat and Qwen-7B-Chat, Q4_K_M) and `llama-cpp-python` with GPU support for full functionality.

## Status

This is a research prototype accompanying an academic paper currently in submission. Interfaces and thresholds (`τ`, `ε`) are tuned for the Qwen 1.5B/7B pair and the evaluated hardware (NVIDIA RTX 50-series, 16GB VRAM, Windows 11) — see the paper for full methodology, limitations, and threshold sensitivity analysis.

## License

MIT — see [LICENSE](LICENSE).
