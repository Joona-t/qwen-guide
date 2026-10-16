# Plan: major improvements to the Qwen guide

> **Status:** DRAFT for Joona's annotation. Add notes inline (e.g. `> JOONA: …`) and I'll revise. **Do not implement yet — awaiting "implement it all".** Evidence for every item is in `research.md`.

## 1. Goal

The guide is written for a 16 GB Mac. The owner wants three changes. First, RTX 3090 and NVIDIA PC owners should be first-class readers: their own path, fit tables, speed expectations, power and dual-GPU advice, and Windows/Linux setup. Second, the guide should be more robust: it needs the KV-cache formula, sampling, chat templates, quant formats, benchmarking, and fixes for the merge seams. Third, it needs a real free learning path, from beginner to advanced, with Karpathy's courses at its core. Every new fact must trace to research_md. Anything uncertain gets labelled as such.

## 2. Proposed table of contents

- **Start here: pick your path** (NEW, WP-1)
- **Part 1 · Run Qwen locally**
  - What you're actually running (EXPANDED: NVIDIA bandwidth figures, fit and tok/s widget)
  - Is local right for your use case? (EXPANDED: VRAM rule alongside the Mac rule; links to the eval harness)
  - Which Qwen? (EXPANDED: VRAM-tier table)
  - Hugging Face (UNCHANGED)
  - Install Ollama (EXPANDED: Mac / Linux systemd / Windows tabs)
  - **Running on an NVIDIA PC: RTX 3060 to 3090 to 5090, and two 3090s** (NEW, WP-2)
  - **Tokens, context and the KV cache** (NEW, WP-5)
  - Thinking mode, plus **Sampling settings** (EXPANDED, WP-5)
  - Other runtimes (EXPANDED: PC runtime table)
  - Faster and smaller (EXPANDED: NVIDIA engines, quant-format table, CUDA speculative decoding)
  - **Benchmark your machine** (NEW, WP-6)
  - Use it from code, plus **Structured output, tools and images** (EXPANDED, WP-7)
  - Keep it on your machine (UNCHANGED)
  - Learn more: running models (EXPANDED, now points into the learning path)
- **Part 2 · Build on it** (EXPANDED, WP-8)
  - Embeddings, RAG and Qdrant (seam fix: one embedder end to end)
  - Shared memory (taxonomy collapsed)
  - Harden it: Security (split out)
  - Route the work (split out; single-24-GB-PC variant; NVIDIA budget preset)
  - Build order (NVIDIA alternatives for stages 1 and 6)
  - Learn more (missing links added)
- **Part 3 · Fine-tune** (EXPANDED, WP-9)
  - Will it fit? (rows for 32 GB, 2×24 GB and 27B)
  - Path B: per-card settings
  - Evaluate honestly (shares the harness from WP-7)
  - Other sections UNCHANGED
- **Part 4 · Abliteration** (UNCHANGED; its Learn-more list gets interpretability additions)
- **Free learning path: from zero to understanding LLMs** (NEW, WP-10)
- **Reference**
  - Troubleshooting (EXPANDED and regrouped, WP-11)
  - **Glossary** (NEW, WP-12)
  - Sources (EXPANDED)
  - Caveats (UNCHANGED)

## 3. Work packages

**WP-1 · Hardware router: "Start here, pick your path"** (P0 · M)
- **Changes:**
  - Add a new section after the hero. A table maps each hardware class to:
    - the memory rule
    - the first model to run
    - the runtime
    - the sections to read
  - Rows: Mac 16, Mac 24-32, Mac 64+, NVIDIA 8-16, RTX 3090 24 GB, dual 3090 48 GB, CPU-only.
  - A second small table routes by goal: chat, code, RAG, agents, fine-tune, learn the theory.
  - Rewrite L107 into two rules. Unified memory: subtract about a quarter. Discrete VRAM: keep about 1 GB plus the KV cache.
  - Retitle the hero to cover all four parts plus learning.
- **Widget:** optional. A path picker that highlights the TOC entries.
- **Evidence:** research §2 fits table; §1 rows 1-3.

**WP-2 · Section: "Running on an NVIDIA PC"** (P0 · L)
- **Changes, in this order:**
  - The cards table, with bandwidths labelled as derived.
  - The "fits in 24 GB" table. Qwen3.8-27B Q4_K_M is the headline pick, at 19 GB with ~4-6 GB left. The A3B MoEs come next, with Q3_K_XL as the long-context option.
  - tok/s expectations. Give the ceiling (bandwidth ÷ size) next to measured numbers: the 7B scoreboard, plus 32-38 t/s for a 27B and ~137 t/s for the 35B-A3B, both labelled secondary. Say that decode falls as context grows.
  - Prefill versus decode, and why prefill matters for RAG.
  - Driver floors: Ollama needs driver 550+.
  - Engine ladder: Ollama, then llama.cpp, then ExLlamaV3/TabbyAPI, then vLLM/SGLang.
  - Power limiting. Cover `nvidia-smi -pl` and that it resets on reboot. A ~250-290 W cap keeps most decode on a 27B, but the effect is model-dependent.
  - Dual 3090: split modes, vLLM TP vs PP, NVLink (both 3090 and 3090 Ti have it), PCIe Gen4, PSU of 850 W or more, and what 48 GB unlocks.
  - Smaller cards: MoE offload with `--n-cpu-moe`.
  - A used-3090 checklist, with memory temperatures marked UNCERTAIN.
  - Checkpoint: "`ollama ps` shows 100% GPU, and you have a measured tok/s."
- **Evidence:** §2 and §3.

**WP-3 · PC variants for install and the Part 1 runtime text** (P1 · S)
- **Changes:**
  - Turn the L210-218 settings into tabs: Mac `launchctl`, Linux `systemctl edit ollama.service` with `Environment=` lines, and Windows (quit the tray app, set user env vars, relaunch).
  - Add a recommended 24 GB NVIDIA set: flash attention, q8_0 KV, a larger num_ctx, and `CUDA_VISIBLE_DEVICES` (UUIDs).
  - Add WSL2 rules: Windows driver only, `cuda-toolkit-12-x`, Windows 11.
  - Add the PC runtime table to "Other runtimes".
  - Make every llama-server snippet pass `--port`, because the default is now 9931.
  - Fix L241: mlx-lm becomes "main Mac path".
- **Evidence:** §3.

**WP-4 · NVIDIA VRAM-fit and tok/s calculator widget** (P0 · M)
- **Changes:**
  - Inputs: hardware preset, model, quant, context, KV type.
  - Hardware presets: 3060 12, 4060 Ti 16, 3090, 3090 Ti, 4090, 5090, 2×3090, plus Mac presets.
  - Model presets come from the fits table.
  - Fit = weights + KV + overhead, compared against usable memory. KV uses per-token constants only where they are verified (Qwen3.6). Other models show "KV estimate unavailable".
  - tok/s ceiling = bandwidth ÷ bytes, shown next to a measured-anchor note.
  - Also add a 3090 preset to the decode widget (`template/widgets/decode.html`), and a 5090 preset.
- **Widget:** NEW `fit.html`. EXPANDED `decode.html`.
- **Evidence:** §2.

**WP-5 · Depth: tokens, context, KV formula, sampling, templates, quant formats** (P0 · M)
- **Changes:**
  - KV formula: 2 × layers × kv_heads × head_dim × ctx × bytes × slots. Work it through for Qwen3.6-27B (64 KiB/token at f16). Explain hybrid attention, and how `OLLAMA_NUM_PARALLEL` multiplies the cache.
  - num_ctx: the FAQ default is 4096, and readers should check theirs with `ollama ps`. Explain silent truncation. Fix L290, since num_ctx works per request.
  - Chat templates: what they are and why a mismatch breaks output.
  - Sampling: temperature, top_p, top_k, min_p and the penalties, and how to set each in Ollama, llama.cpp and the OpenAI client. Sampling values come from the official model card only, after it is verified at implementation time.
  - A quant-format-by-GPU table.
  - Fix the 4B size seam (L25 vs L801).
- **Evidence:** §2 KV, §3 formats, §1 seams.

**WP-6 · Benchmark your machine** (P0 · S)
- **Changes:**
  - `ollama run … --verbose`.
  - The API duration fields, and how to turn them into tok/s.
  - `llama-bench -p 512 -n 128 -ngl 99 -fa on`, with `-d` for depth.
  - `ollama ps` and `nvidia-smi` for offload checks.
  - A fill-in results table.
  - Say plainly that the 3090 speeds for gpt-oss, Gemma 4, GLM, Coder and Glimmer are unmeasured. This section is where readers fill that gap.
- **Evidence:** §3 llama-bench; §6.

**WP-7 · Structured output, tools, images, eval harness** (P1 · M)
- **Changes:**
  - A JSON-schema call, using Ollama `format` and a llama.cpp grammar.
  - A tool-calling loop of about 30 lines over `/v1`, with an allowlist.
  - Image input through the Ollama CLI and API, and llama.cpp mmproj.
  - An N-models × M-prompts harness that writes a CSV, reused by Part 3's eval.
  - A CUDA speculative-decoding command (llama-server draft model). Speedup figures stay labelled per source.
- **Evidence:** §1; §2 drafter note (DFlash/MTP GGUFs exist).

**WP-8 · Part 2 PC variant and seam fixes** (P1 · M)
- **Changes:**
  - Pick one embedder end to end. Recommended: Qwen3-Embedding-0.6B at 1024 dims across L395, 471, 543-547, 680 and 709-710. Add a dims check to the Qdrant snippet.
  - Add a single-24-GB-PC budget: a 27B chat model plus the embedder plus the reranker on the GPU, with the vector store in system RAM.
  - Add an NVIDIA preset to `budget.html`.
  - Point out that a 27B clears the tool-calling cliff the guide cites at L688.
  - Give Linux and Windows notes for the sqlite-vec gotcha.
  - Docker examples use `--gpus all`.
  - Give stage 6 an Unsloth alternative.
  - Split "harden" into Security and Route the work.
  - Add the missing Learn-more links: Qdrant docs, MCP spec, OWASP Agentic, Contextual Retrieval, RAGAS. Each must be verified live first.
- **Widget:** EXPANDED `budget.html`.
- **Evidence:** §1.

**WP-9 · Part 3 NVIDIA rows** (P1 · S)
- **Changes:**
  - Extend the fit table with the Unsloth figures: Qwen3.5 bf16 LoRA (4B 10, 9B 22, 27B 56 GB), and the general QLoRA/LoRA16 minimums (27B 22/64, 70B 41/164).
  - Add a row for 2×24 GB using `device_map='balanced'`, labelled UNCERTAIN.
  - Note that a 27B bf16 LoRA (56 GB) does not fit on 48 GB.
  - Add per-card comments to the script.
- **Evidence:** §2 fine-tuning.

**WP-10 · Free learning path** (P0 · M)
- **Changes:** a new section with a goal table, then tracks. Every entry uses one format: name, format, level, time, what it teaches, free status. The section is dated "checked 2026-10".
  - **Track 0, orientation:** Karpathy Intro to LLMs, Deep Dive and How I use LLMs; 3Blue1Brown "LLMs explained briefly"; Transformer Explainer.
  - **Track 1, foundations:** micrograd and the first lectures of Zero to Hero; fast.ai; MIT 6.S191; Google MLCC's LLM module.
  - **Track 2, build an LLM:** Zero to Hero (Let's build GPT, the Tokenizer lecture); Let's reproduce GPT-2 with build-nanogpt; nanochat (nanoGPT marked deprecated); minbpe; Raschka's repo (book marked paid); Illustrated Transformer; LLM Visualization; the Vaswani paper; 3Blue1Brown Ch5-7.
  - **Track 3, run and serve:** Ollama, llama.cpp, vLLM and Unsloth docs; Labonne's quantization notebooks.
  - **Track 4, fine-tune and post-train:** Hugging Face LLM Course; smol course (units 1-4); Labonne llm-course; Unsloth docs.
  - **Track 5, agents:** Hugging Face Agents Course; Open-Source AI Cookbook.
  - **Track 6, university courses:** CS336; CS224N (2024 videos); MIT 6.S191.
  - **Stay current:** Lil'Log, Simon Willison's local-llms tag, r/LocalLLaMA (flagged as anecdotal).
  - **Also:** point each part's Learn-more list at its track, remove the duplicate Hugging Face course entry, and leave LLM101n out. Mark DeepLearning.AI as free only during the beta.
- **Evidence:** §4, §5.

**WP-11 · Troubleshooting** (P1 · S)
- **Changes:**
  - Regroup the table into Run, Build, Fine-tune, Abliteration and Platform.
  - New rows:
    - CPU/GPU split in `ollama ps`
    - silent num_ctx truncation
    - think loops
    - OOM at long context
    - WSL GPU missing (Windows driver only)
    - Windows env var ignored (quit the tray app)
    - systemd `OLLAMA_HOST` override
    - second GPU idle (`CUDA_VISIBLE_DEVICES`, `-ts`)
    - Docker without `--gpus all`
    - llama-server on an unexpected port (9931)
    - GPU lost after Linux suspend (`rmmod`/`modprobe nvidia_uvm`)
- **Evidence:** §3.

**WP-12 · Glossary, sources, link check** (P2 · S)
- **Changes:**
  - About 40 glossary terms, each linked to where it is first used.
  - Add the new sources to Sources.
  - Verify or fix L1159 (`QwenLM/Qwen3.8`).
- **Evidence:** §1, §6.

## 4. Sequencing and definition of done

**Order:**
- **Wave 1, parallel:** WP-2, WP-5 and WP-10. They hold the facts everything else depends on.
- **Wave 2:** WP-1, which needs WP-2's anchors, and WP-4, which needs WP-2's tables and WP-5's formula.
- **Wave 3, parallel:** WP-3, WP-6, WP-7, WP-8 and WP-9.
- **Wave 4:** WP-11 and WP-12, which reference everything before them.

**Definition of done for every WP:**
- The guide's existing check suite passes. It is 223 checks today, and new widgets add their own checks.
- The page renders without overlap or horizontal scroll at 1400 px and 360 px, in both themes.
- Every new number cites a URL from research_md and keeps its label: measured, derived, secondary or UNCERTAIN.
- No refuted or downgraded claim appears unlabelled.
- Every new link is fetched live just before publishing.
- New code snippets are run once on a real machine, or marked "untested".
- Claude logs each WP in `BUGS_AND_ITERATIONS.md`, commits with the co-author trailer, and adds a line to NOW.md's Shipped list on deploy.

## 5. Owner decisions (max 4)

1. **Placement of the learning path.** Should it be a front-of-guide "Part 0", or a Reference section linked from the router? The recommendation is Reference plus the router link, so Part 1 still opens on running a model.
2. **Measured 3090 numbers.** Should the guide run its own llama-bench on a real 3090 to fill the unmeasured rows, or ship them as estimates only? A measurement needs hardware access.
3. **Default embedder.** Qwen3-Embedding-0.6B at 1024 dims (recommended), or keep nomic at 768 and downgrade the "default" claim?
4. **Scope of non-NVIDIA PCs.** AMD, Intel Arc and CPU-only: a router row only (recommended for now), or a full subsection, which would need a new research pass?

## 6. Status

Do not implement yet. Awaiting "implement it all".
