# Research: Qwen guide audit and evidence base (checked 2026-10-16)

> **Status:** evidence base for `plan.md`, compiled 2026-10-16 by an 11-agent audit (4 auditors over `source/qwen-guide.md`, 3 web scouts, 3 adversarial skeptics, 1 planner). 79 claims verified, 8 refuted, 31 downgraded. Spot-checked by hand afterwards: the llama.cpp RTX 3090 scoreboard (161.89 t/s tg128 with FA, Llama 2 7B Q4_0), the llama-server default port (9931), nanoGPT's deprecation in favour of nanochat, and 16 key resource links (all HTTP 200).

Source: `/Users/darkfire/qwen-guide/source/qwen-guide.md` (1,177 lines). Four auditors looked at it: hardware, depth, learning resources, and structure. Three scout-and-skeptic pairs checked the facts. Only skeptic-verified facts appear below. Anything marked UNCERTAIN must be labelled as uncertain in the guide, or left out.

## 1. Audit findings (deduplicated)

| Sev | Gap | Where | Evidence |
|---|---|---|---|
| P0 | No "pick your path" router by hardware or goal | hero L1-17, run-basics, use-case L107 | L7-11 only list the parts. L107 says "Subtract a quarter for the system", which is a unified-memory rule and wrong for discrete VRAM. |
| P0 | No NVIDIA or RTX 3090 inference path | run-basics L30, install L190-219, faster L246-279 | "3090" appears once, at L1101 in Part 4. "VRAM", "multi-GPU" and "4090/5090" appear 0 times outside the widget. L30 gives Mac figures only. |
| P0 | No "what fits in N GB of VRAM" table for inference | models L117-137 | The only NVIDIA-by-VRAM table is for LoRA training (L815-823). |
| P0 | KV cache never quantified | L25, L290, L276 | "Add a gigabyte or two…" There is no formula, and the hybrid-attention effect is missing. |
| P0 | No "benchmark your machine" section | Part 1; caveat L724 | llama-bench, `--verbose` and eval_count appear 0 times. Prefill is mentioned once, as a warning (L256). |
| P0 | No free learning path; Karpathy gets one bullet | learn-run L304-313 and the other Learn-more lists | 3Blue1Brown, nanoGPT, Stanford and fast.ai appear 0 times. |
| P1 | The Ollama settings block is Mac-only | install L210-218 | Six `launchctl` lines, then one sentence for Linux and Windows. |
| P1 | The faster section covers Mac engines only | L250-279 | Nothing on EXL3 or TabbyAPI. vLLM is framed as "not for a laptop". FP8 and NVFP4 are never tied to GPU generations. |
| P1 | Part 2 assumes a 16 GB Mac | L341, L421, L534, L676-682, L708-719 | The budget widget is Mac-only. Build stage 6 offers mlx-lm only. |
| P1 | Seam: the "default" embedder contradicts the code | L349 vs L395, L471, L543-547, L709-710 | The prose says Qwen3-Embedding-0.6B (1024 dims). Every code sample uses nomic at 768 dims. |
| P1 | Sampling gets one line | L232 | min_p and presence_penalty appear 0 times. Nothing on how to set sampling in each runtime. |
| P1 | Tokens, context and chat templates are never explained | L23, L290, L1149 | L290 says context is "per model, not per request". That contradicts the Ollama FAQ (num_ctx works per request). |
| P1 | No decision guide for GPU quant formats | L147-152, L278 | EXL2/EXL3 appear 0 times. AWQ, GPTQ, FP8 and NVFP4 appear once, in passing. |
| P1 | Structured output and tool calling are never demonstrated | api L280-290, L686-700 | No `tools` array and no tool-calling loop. |
| P1 | Model choice has no eval harness | L112 | "Write ten to thirty real inputs…" but no tooling. |
| P1 | Fine-tune table stops at 24 GB and has no 27B row | L815-823, L943 | No rows for 32 GB, 2×24 GB or a 27B on 24 GB. |
| P1 | Troubleshooting has no NVIDIA, WSL or Windows rows, and few inference rows | L1135-1153 | 14 rows; about 8 are about training or export. |
| P1 | No glossary | whole guide | Terms like KV, A3B, DWQ, NVFP4 and GRPO are defined once at most. |
| P2 | Dual-GPU (2×3090) absent | – | No tensor split, power limiting, PCIe or NVLink. |
| P2 | Seam: two sizes for the same 4B model | L25 vs L801 | "about 9 GB / 2.5 GB" vs "about 8 GB / 2 GB". |
| P2 | No how-to for images, Mac-only speculative decoding, and AMD/Intel/CPU-only get one line each | L119, L260, L190 | No mmproj example. No llama.cpp draft command. |
| P2 | Five unlinked Part 2 resources and a suspect source link | L733-742, L1159 | Qdrant docs, MCP spec, OWASP Agentic, Contextual Retrieval and RAGAS have no link. `github.com/QwenLM/Qwen3.8` is unverified. |

Strengths to keep: the bandwidth rule (L30, L34), the honest ◆/📝 evidence labels, the security depth, the RAG depth, the fine-tune eval discipline, and the per-part Learn-more lists. Learn-abl (L1121-1131) is the model to copy for the others.

## 2. RTX 3090 and NVIDIA PCs (skeptic-corrected)

### Cards

| Card | Arch (CC) | VRAM / bus | Bandwidth GB/s | Power / PSU | FP8 HW | FP4 HW | NVLink | llama.cpp 7B Q4_0 tg128 t/s (FA on) |
|---|---|---|---|---|---|---|---|---|
| RTX 3060 12GB | Ampere 8.6 | 12 GB / 192-bit (page not fetched) | 360 (derived) | 170 W UNCERTAIN | No | No | No | 76.9 |
| RTX 4060 Ti | Ada 8.9 | 16 or 8 GB / 128-bit | 288 (derived) | 160/165 W, 550 W | Yes | No | No | 64.0 (variant not stated) |
| RTX 3090 | Ampere 8.6 | 24 GB GDDR6X / 384-bit | 936 (derived) | 350 W, 750 W (FE) | No (W8A16 weight-only) | No | Yes | 161.9; 177.8 (build 10809, Sep 2026); 155.4 at 390 W, 138.3 at 250 W |
| RTX 3090 Ti | Ampere 8.6 | 24 GB / 384-bit | 1008 (derived, UNCERTAIN) | 450 W, 850 W | No | No | Yes (NVIDIA page) | 172.3 |
| RTX 4090 | Ada 8.9 | 24 GB / 384-bit | 1008 (derived) | 450 W UNCERTAIN | Yes | No | No | 189.0 |
| RTX 5090 | Blackwell 12.0 | 32 GB GDDR7 / 512-bit | 1792 (derived) | 575 W, 1000 W | Yes | Yes | No | 300.4 (245.0 at a 400 W cap) |

Sources:
- https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
- https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
- https://developer.nvidia.com/cuda-gpus
- https://github.com/ggml-org/llama.cpp/discussions/15013
- https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/

The bandwidth figures are bus width × data rate. No NVIDIA page lists them, so label them as derived.

### What fits on one 3090 (24 GB)

| Model | Type | 4-bit size | Left after weights | Context note | Speed |
|---|---|---|---|---|---|
| gpt-oss-20b | MoE, 20.9B total | 14 GB (Ollama) | ~9-10 GB | Large; 128K UNCERTAIN | Not measured |
| Muse Glimmer 30B Q4_K_M | Dense ~28B, 3/4 SWA, 2 KV heads | 16.8 GB (vendor's 24 GB pick) | ~6 GB | Vendor: ~17/19/20 GB for text / +vision / +drafter | Ceiling ~55 t/s (derived) |
| Qwen3.8-27B Q4_K_M | Dense, arch qwen35 | 19 GB (ggml-org) | ~4-6 GB | ~64K f16 or ~128K q8_0 KV, if the layout matches 3.6 (UNCERTAIN) | Ceiling ~49; Qwen3.6-27B Q3_K_XL measured 32-38 (club-3090, secondary) |
| Gemma 4 31B Q4_K_M | Dense, SWA | 18.3 GB | ~4 GB minus ~3.6 GB SWA cache | Tightest: 22.3 GB at 16K (user report); use `-np 1` and q8_0 KV | Ceiling ~51 (derived) |
| GLM-4.7-Flash Q4_K_M | MoE 30B-A3B, deepseek2/MLA | 18.3 GB | ~4-5 GB | Compact KV (UNCERTAIN) | Not measured |
| Qwen3-Coder-30B-A3B | MoE, 3B active | 18.6 GB Q4_K_M / 17.7 UD-Q4_K_XL | ~4-6 GB | Moderate (UNCERTAIN) | Not measured |
| Qwen3.6-35B-A3B | MoE, hybrid, 2 KV heads | 22.1 GB Q4; 16.8 GB UD-Q3_K_XL | ~0.5-1.5 GB at Q4 | 20 KiB/token (derived); use Q3_K_XL or `--n-cpu-moe` | 136.8 t/s at 370 W, 114.6 at 210 W (club-3090, secondary) |

Sources:
- https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main
- https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main
- https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/discussions/2
- https://ollama.com/library/gpt-oss:20b
- https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
- https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/tree/main
- https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
- https://github.com/noonghunna/club-3090/blob/master/docs/HARDWARE.md

The KV formula, verified for Qwen3.6-27B: only the 16 full-attention layers keep a cache. 16 × 4 KV heads × 256 head dim × 2 (K and V) × 2 bytes = 64 KiB per token at f16. That is 2 GiB per 32K tokens.

### Two 3090s (48 GB)

- **70B dense at Q4_K_M:** 16.29 tg / 393.9 pp tok/s, falling to 14.34 tg at 8K. 2×4090 gets 19.06 (XiongjieDai, May 2024, secondary).
- **Qwen3.6-35B-A3B Q8_0:** 36.9 GB, verified.
- **Qwen3.8-27B Q8_0:** 28.6 GB, which leaves ~18 GB for KV.
- **Long context:** decode drops sharply. Dual 3090 with a 27B Q6 runs ~22.8 t/s at low context and 4.0 at 95K (sanj.dev, secondary).
- **llama.cpp:** `--split-mode none|layer (default)|row|tensor (experimental)`, plus `--tensor-split` and `--main-gpu`. The NVLink benefit for layer split is UNCERTAIN.
- **vLLM:** `--tensor-parallel-size 2` runs on Linux only. Without NVLink, the docs advise pipeline parallelism.
- **ExLlamaV3:** supports tensor- and expert-parallel inference.
- **PCIe:** Gen4 is recommended. Gen3 hosts can lose 30-40% on TP=2 (secondary, and CPU age is a confound).
- **Power:** two cards capped at 330 W draw ~660 W, so use an 850 W or larger PSU. A 250 W cap kept ~99% of decode on a 27B (jeanfbrito, single run) but cost ~11% on the 7B bench (RodriMora). The effect depends on the model. `nvidia-smi -pl` resets on reboot, so it needs persistence mode or a systemd unit.
- **Fine-tuning (Unsloth):**
  - Qwen3.5 bf16 LoRA needs 4B 10 GB, 9B 22 GB, 27B 56 GB. Unsloth does not recommend QLoRA on Qwen3.5.
  - Its general QLoRA/LoRA16 minimums include 27B 22/64 GB and 70B 41/164 GB. The minimum CUDA capability is 7.0.
  - Multi-GPU uses `device_map='balanced'`. Whether that works well for a 27B is UNCERTAIN.

Sources:
- https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
- https://github.com/turboderp-org/exllamav3
- https://jeanfbrito.github.io/posts/rtx-3090-power-limit-sweet-spot/
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

## 3. PC runtimes and formats (corrected)

| Runtime | OS | Best for | Multi-GPU | Formats | Latest |
|---|---|---|---|---|---|
| Ollama | Win 10 22H2+ (driver 551.61+), Linux, macOS | Easiest start | Automatic; `CUDA_VISIBLE_DEVICES` (use UUIDs) | GGUF | v0.40.5 (10-14) |
| llama.cpp | Win prebuilt CUDA 12.4/13.4 (+cudart zip); Ubuntu CUDA 12.8/13.4 | Control, benchmarking, `--n-cpu-moe` | `-sm`, `-ts`, `-mg` | GGUF | b11696 (10-16) |
| LM Studio | Win x64 (AVX2)/ARM, Linux AppImage, macOS | GUI + server (examples use :1234/v1) | Per-GPU on/off (v0.3.14); CUDA tensor parallel (v0.4.15) | GGUF; MLX on Mac | UNCERTAIN |
| vLLM | Linux only (Windows via WSL2) | Batching, many agents | TP / PP | AWQ, GPTQ, Marlin FP8/FP4 weight-only on Ampere, INT8, GGUF | v0.31.0 (10-05) |
| SGLang | Linux, CUDA 13 only | Serving, structured output | `--tp-size` | AWQ/GPTQ/FP8 (Ampere UNCERTAIN) | v0.5.21 |
| ExLlamaV3 + TabbyAPI | Win (triton-windows), Linux | Quality per GB on 24 GB; API :5000 | TP, EP, CPU MoE (Docker `--shm-size`) | EXL3 | v1.6.2 |
| KoboldCpp | Win exe, Linux binary | Single file; :5001/v1 | UNCERTAIN (`--usecuda`, `--gpulayers`) | GGUF | v1.123.1 |
| text-generation-webui | Win/Linux/macOS portable | Loader switcher; :7860 | Per loader | GGUF, EXL3, Transformers | v4.9 (May 2026, slowing) |
| Open WebUI | Docker | Front end; :3000 | n/a | n/a | v0.12.3 |

Key facts from these sources:
- **llama-server default port.** It now defaults to 127.0.0.1:9931, not 8080. Every snippet should pass `--port` explicitly.
- **Ollama on Linux.** Set env vars with `systemctl edit ollama.service` (`Environment=` lines), then `daemon-reload` and restart.
- **Ollama on Windows.** Quit the tray app, set user env vars, then relaunch.
- **Ollama config.** `ollama ps` shows the CPU/GPU split. `OLLAMA_KV_CACHE_TYPE` needs flash attention. The FAQ gives the default context as 4096; tell readers to check theirs with `ollama ps`.
- **WSL2.** Install only the Windows driver, plus `cuda-toolkit-12-x` inside WSL. Never install the `cuda` or `cuda-drivers` packages there. Docker supports only `--gpus all`. Windows 11 needs no Insider build.
- **Peer-to-peer.** `GGML_CUDA_P2P` usually needs workstation drivers, so it is likely a no-op on a 3090.

Sources:
- https://docs.ollama.com/windows
- https://docs.ollama.com/gpu
- https://docs.ollama.com/faq
- https://docs.ollama.com/linux
- https://github.com/ggml-org/llama.cpp/releases
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- https://docs.vllm.ai/en/latest/features/quantization/
- https://github.com/theroyallab/tabbyAPI
- https://docs.sglang.io/get_started/install.html
- https://github.com/LostRuins/koboldcpp
- https://github.com/oobabooga/text-generation-webui
- https://docs.openwebui.com/getting-started/quick-start/
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://lmstudio.ai/docs/app/system-requirements

### Quant formats by GPU generation

| Format | Ampere (3090) | Ada (4090) | Blackwell (5090) |
|---|---|---|---|
| GGUF K/IQ | Yes | Yes | Yes |
| EXL3 | Yes in practice (one 3090 user report) | UNCERTAIN | UNCERTAIN |
| AWQ / GPTQ-Marlin / INT8 W8A8 / bitsandbytes | Yes | Yes | UNCERTAIN (no vLLM column) |
| FP8 W8A8 | No; weight-only W8A16 | Yes | Yes |
| NVFP4 / MXFP4 | Fallback only (vLLM Marlin FP4; llama.cpp generic kernels, medium confidence) | Fallback only | Native FP4 tensor cores |
| KV cache quant | llama.cpp `-ctk/-ctv` q8_0/q4_0…; Ollama `OLLAMA_KV_CACHE_TYPE`; EXL3 2-8 bit | Same | Same |

## 4. Free learning resources (corrected)

| Resource | URL | Length / level | Free status |
|---|---|---|---|
| Karpathy, Neural Networks: Zero to Hero | https://karpathy.ai/zero-to-hero.html | 8 lectures, 14h32m; beginner-intermediate | Public YouTube (the page never says "free") |
| Karpathy, Let's build GPT | https://www.youtube.com/watch?v=kCc8FmEb1nY | 1h56m; intermediate | Free |
| Karpathy, Let's reproduce GPT-2 (124M) | https://youtu.be/l8pRSuU81PU + https://github.com/karpathy/build-nanogpt | ~4h (secondary) | Free video; README puts the repro at ~1h and ~$10 of compute |
| Karpathy, Intro to Large Language Models | https://www.youtube.com/watch?v=zjkBMFhNj_g | ~1h (from title); beginner | Free |
| Karpathy, Deep Dive into LLMs like ChatGPT | https://www.youtube.com/watch?v=7xTGNNLPyMI | 3h31m (unconfirmed); general | Free |
| Karpathy, How I use LLMs | https://www.youtube.com/watch?v=EWvNQjAaOHw | Length UNCERTAIN | Free |
| nanochat | https://github.com/karpathy/nanochat | ~2h on 8×H100 (~$48); 1 GPU ~8× slower | MIT code; compute costs money |
| nanoGPT (deprecated Nov 2025) | https://github.com/karpathy/nanoGPT | – | MIT; README points to nanochat |
| llm.c / minbpe / micrograd | https://github.com/karpathy/llm.c · /minbpe · /micrograd | Advanced / intermediate / beginner | MIT |
| 3Blue1Brown Deep Learning Ch5-7 + "LLMs explained briefly" | https://www.3blue1brown.com/topics/neural-networks | Lengths UNCERTAIN; beginner | Free |
| Hugging Face LLM Course | https://huggingface.co/learn/llm-course | 12 chapters, ~6-8h each | Free, no ads |
| Hugging Face smol course | https://github.com/huggingface/smol-course | Units 1-4 released (5-6 unconfirmed) | Free, "no paid services" |
| Hugging Face Agents Course | https://huggingface.co/learn/agents-course | Units 0-4 + 3 bonus | Free, including the certificate |
| Hugging Face Open-Source AI Cookbook | https://huggingface.co/learn/cookbook | Varies | Free |
| fast.ai Practical Deep Learning | https://course.fast.ai/ | 9 × ~90 min; Part 2 30h+ | Free |
| Stanford CS336 (Spring 2026) | https://cs336.stanford.edu/ (playlist PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) | Full quarter; advanced | Free lectures; full assignments need big GPUs |
| Stanford CS224N | https://web.stanford.edu/class/cs224n/ (2024 playlist PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) | Full quarter | 2024 videos free; 2026 videos enrolled-only |
| MIT 6.S191 (2026) | https://introtodeeplearning.com/ | ~9 weeks; Lab 3 fine-tunes an LLM | Free |
| Raschka, LLMs-from-scratch | https://github.com/rasbt/LLMs-from-scratch | Ch1-7 + App. A-E; runs on a laptop | Code free (Apache-2.0); book paid |
| Labonne, llm-course | https://github.com/mlabonne/llm-course | Self-paced, 3 tracks | Free; Handbook paid |
| Alammar, The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | Beginner-intermediate | Free (CC BY-NC-SA) |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | 15 pages | Free |
| Transformer Explainer | https://poloclub.github.io/transformer-explainer/ | Minutes; beginner | Free to use |
| LLM Visualization (Bycroft) | https://bbycroft.net/llm (repo github.com/bbycroft/llm-viz; /llm path UNCERTAIN) | Minutes-1h | Free (MIT) |
| Lil'Log | https://lilianweng.github.io/ | Long posts | Free |
| Simon Willison, local-llms tag | https://simonwillison.net/tags/local-llms/ | 165 posts | Free |
| Google ML Crash Course | https://developers.google.com/machine-learning/crash-course | LLM module ~45 min | Price not stated (UNCERTAIN) |
| DeepLearning.AI short courses | https://www.deeplearning.ai/short-courses/ | 49m-2h each | Free only "for a limited time during beta" |
| llama.cpp / Ollama / vLLM / Unsloth docs | github.com/ggml-org/llama.cpp · docs.ollama.com · docs.vllm.ai · unsloth.ai/docs | Reference | Free (MIT / MIT / Apache-2.0 / Apache-2.0 core + AGPL Studio) |
| r/LocalLLaMA | https://www.reddit.com/r/LocalLLaMA/ | Forum | Free (not fetched; a lead only) |

## 5. Refuted or downgraded (keep out of the guide as stated)

Refuted:
- "The 3090 is the last GeForce card with NVLink." The 3090 Ti also has it.
- "sanj.dev shows dual layer-split is not faster than one card." The article has no single-card numbers.
- "Google MLCC ~15h of LLM content." The LLM module is ~45 min.
- "DeepLearning.AI videos are free as standing policy." Free access is temporary, during the beta.
- "llm.c reproduces GPT-3." Its focus is the GPT-2/GPT-3 miniseries.
- "`lms server start --port 1234` is documented." It is not.
- The LM Studio bug #2234 has nothing to do with the 3090.
- The Ollama Windows models path `%HOMEPATH%\.ollama` is unverified; cite `OLLAMA_MODELS` instead.

Downgraded (label as estimate or secondary):
- "No speed gain from FP4 emulation" is an inference.
- All bandwidth figures are derived.
- All club-3090 numbers come from a community repo with single contributors. Its NVLink 1.6-1.8× claim contradicts its own 10-15% claim.
- jeanfbrito's power sweep is a single run, through llama-swap.
- The WSL2 VRAM overhead and TDR fix are vLLM-specific and secondary.
- The Gemma 4 SWA cache figure is a user report.
- Used-3090 memory temperatures and throttle points are unconfirmed.
- `GGML_CUDA_P2P` is likely a no-op on consumer cards.
- `CMAKE_CUDA_ARCHITECTURES=86` is derived from the docs, not stated in them.
- EXL3 on a 3090 rests on one user report.
- minbpe matches GPT-4 tokenization only through its wrapper.
- Durations for Deep Dive, the GPT-2 repro, Intro to LLMs and 3Blue1Brown are secondary.
- nanochat's "chat UI" is UNCERTAIN.
- LLM101n is **not released**. Never link it as a course.

## 6. Still uncertain

- The Qwen3.8-27B layer layout. The KV-per-token figure is proven only for 3.6.
- No measured 3090 decode for gpt-oss-20b, Gemma 4 31B, GLM-4.7-Flash, Qwen3-Coder-30B or Muse Glimmer.
- Ollama's default context on current builds.
- Speedups on Blackwell for AWQ, GPTQ and bitsandbytes.
- SGLang FP8 and AWQ on Ampere.
- KoboldCpp multi-GPU.
- The LM Studio version.
- The EXL2 legacy format in v3.
- The gpt-oss-20b active parameter count and context length.
- Unsloth multi-GPU readiness.
- The `QwenLM/Qwen3.8` repo URL (L1159).
- smol course units 5-6.
- The r/LocalLLaMA wiki and the Hugging Face Discord.
