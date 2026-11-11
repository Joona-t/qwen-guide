# Plan: one guide, two tracks: deeper for technical readers, easier for newcomers (2026-11-11)

> **Status:** PROPOSED, not implementing yet. It waits for Joona's annotations and "implement it all". Evidence: `research.md` top section, "Research: two readers, one guide (checked 2026-11-11)", cited below as §I-§IV.

## 1. Goal

Joona, 2026-11-11: *"We want to start getting deeply technical and catering towards technical people, and not only technical people, we want to also help bridge the gap between non-technical people and give them the tools and information they need so that they can go out there and acquire this knowledge."*

This pass delivers three things:
- **For the technical reader:** our own measured numbers from the two-3090 Ubuntu rig, and the mechanisms behind them (roofline, attention kernels, all-reduce, speculative acceptance, quantization error), each with a profiler they can run themselves.
- **For the newcomer:** a path from "I have a computer" to "a local model answered me" that assumes nothing, plus a checkpoint at the end of every Part 1 section.
- **For both:** the same sections, not two documents. Detail they don't need yet folds away, so neither reader wades through the other's material.

**Frontier references** (§IV):
- **Technical:** *How To Scale Your Model* (inference chapter), the *Ultra-Scale Playbook*, and *Making Deep Learning Go Brrrr*. None of them is measured on consumer Ampere cards with a hybrid-attention 27B.
- **Beginner:** Ollama's quickstart and LM Studio's onboarding.

**Concrete acceptance criteria** are in §5.

## 2. The structure: one stream, two depths

```
Section (unchanged order)
 ├─ Goal line          "You'll have: … · about N min"            ← newcomer
 ├─ Main text          plain words, first-use definitions        ← both
 ├─ ▸ Under the hood   <details>, closed by default              ← technical
 │    mechanism · derivation · profiler walkthrough · our bench rows
 └─ ✅ Checkpoint       "You should now see X. If Y → Troubleshooting"
```

- **Beginner track:** the main text with every Under-the-hood block closed. It is about the guide as it reads today, minus the asides that currently interrupt it, which move into blocks.
- **Under-the-hood track:** open the blocks. A "Show all Under the hood" toggle in the rail opens every block at once for readers who want the deep version. It is per viewer, stored in `localStorage` with try/catch, like the theme choice.
- **Why not two documents:** two copies drift, and the freshness check (`scripts/freshness.py`) and `verify.mjs` only guard one source. The `<details>` blocks keep one source of truth (§III).

## 3. Work packages

Effort is in focused sessions: S ≈ under 1, M ≈ 1-2, L ≈ 3+. Priority: P0 ships first and unblocks others. P1 is the core of the pass. P2 is worth it if time allows.

| # | Work package | What changes | Evidence | Effort | Priority |
|---|---|---|---|---|---|
| **W1** | **Rig bench harness + first measured rows** | `scripts/bench_rig.sh` (runs on the Ubuntu 2×3090 box) runs a fixed matrix and writes `bench/2026-11-DD-*.jsonl` with llama.cpp build hash, vLLM version, driver, power cap, PCIe link width/gen (`nvidia-smi -q`), and RAM. The matrix: (a) `llama-bench` Qwen3.8-27B at Q4_K_M, UD-Q4_K_XL, Q6_K and Q8_0; `-sm layer` against `-sm tensor`; `-d 0,32768,131072`. (b) `vllm bench serve` with AWQ-INT4, TP=2 against PP=2, concurrency 1/4/16. (c) MTP on/off for both engines. (d) Power-cap sweep 350→220 W. Every 3090 claim the guide marks "ceiling", "secondary" or "untested here" gets our row beside it. Our row does not replace the cited one; the ceiling stays as the explanation. | §I1 | L (mostly wall-clock on the rig) | **P0** |
| **W2** | **Build support for two depths** | `build.py`: a `<!-- deep title="…" -->…<!-- /deep -->` marker renders `<details class="deep">`, and a `✅` callout tag renders a checkpoint. A rail toggle opens or closes every deep block. CSS for both across all themes. `verify.mjs` asserts: deep blocks balanced, every Part 1 section has a checkpoint, and every internal `#anchor` (glossary included) resolves to a real section id. | §II4, §III, §I7 | M | **P0** |
| **W3** | **"Start from zero" path** | A new short section before "What you're actually running": what a terminal is and how to open it on Mac, Windows and Ubuntu; the no-terminal route (LM Studio download → search Qwen3.5-4B → chat); then the Ollama route with the *expected output* shown after each command. Ends with "✅ You'll have: a model answering you, on your machine, offline." Move the "Pick your path" table *after* the four-paragraph "What you're actually running" explainer, or give it a plain-words column. | §II1, §II3 | M | **P0** |
| **W4** | **Define-on-first-use pass + glossary upgrade** | Every term in the §II1 table gets a one-clause definition where it first appears, linked to the glossary. Add the about 15 missing glossary terms (bpw, KLD, perplexity, PagedAttention, continuous batching, prefix caching, MTP, imatrix, pipeline parallel, all-reduce, PCIe, terminal, llama.cpp, Ollama, Hugging Face). Fix the three stale glossary anchors. | §II1, §I7 | M | P1 |
| **W5** | **Checkpoints and goal lines for Part 1** | Each of the Part 1 sections opens with "You'll have: … · about N min" and closes with a ✅ Checkpoint naming the observable result (a command's output, an `ollama ps` line, a tok/s number) and the Troubleshooting row to read if it fails. W2's verify check enforces it. | §II2 | M | P1 |
| **W6** | **Under the hood: the roofline** | A deep block in "Prefill versus decode". It covers arithmetic intensity, the 3090 ridge point (GA102 peak from NVIDIA's whitepaper, cited), and why batch-1 decode is bandwidth-bound and prefill compute-bound. It is checked against our W1 pp/tg rows. A small widget: slide the batch size and watch decode move toward the ridge (this is why vLLM batching is "free"). The math goes on `window`, tested in `verify.mjs` like FineTuneMath. | §I3 | M | P1 |
| **W7** | **Under the hood: profiling two GPUs** | A deep block in "Two RTX 3090s". (1) `nvidia-smi dmon -s pucm` during `-sm layer` against `-sm tensor`: see the cards take turns against working together. (2) `nsys profile --trace=cuda,nvtx` on one decode step: find the all-reduce gaps. (3) `nccl-tests all_reduce_perf` for our PCIe latency. Includes our screenshots or trimmed text output from the rig, and fixes the "GPU utilisation should be high" trap at line 984. | §I2, §I4 | M | P1 |
| **W8** | **Under the hood: what tensor parallelism costs** | A deep block in "vLLM vs Ollama → What changes with two GPUs". Megatron's two all-reduces per layer; NCCL's `2(n−1)/n` cost; the derived Qwen3.8-27B figures (10 KiB × 128 calls per token, about 40 µs of wire time); then **our measured per-call latency** from W7 explains why 2 cards give about 1.5-1.7×, not 2×. Also explains where NVLink would help, and why the claims vary. | §I4 | S (after W7) | P1 |
| **W9** | **Under the hood: speculative decoding acceptance** | A deep block in "Speculative decoding on a GPU". Leviathan's expected-tokens formula with a worked α/γ table, MTP and draft heads against draft models (Medusa, EAGLE, Gloeckle, DeepSeek-V3), how to read the acceptance rate from llama-server and vLLM logs, and **our measured α and tok/s** for MTP and DFlash on Qwen3.8-27B from W1(c). Optional widget: α and γ sliders → expected tokens per pass. | §I5 | M | P1 |
| **W10** | **Under the hood: quantization error** | A deep block in "Quantization, from first principles": Δ = R/(2^b−1), MSE = Δ²/12, why one outlier spoils a block, then outliers (LLM.int8), SmoothQuant, GPTQ and AWQ in one paragraph each. A widget: 32 weights in a block, drag one outlier, watch the other 31 weights' error. **Our measured KLD table** for Qwen3.8-27B (Q8_0 reference; Q6_K, Q4_K_M, UD-Q4_K_XL, UD-Q3_K_XL) fills the "nobody has tested" line for the GGUF side. | §I6 | M-L | P1 |
| **W11** | **Under the hood: attention kernels** | A deep block in "Tokens, context and the KV cache": standard attention's N×N score matrix, how FlashAttention tiles it through SRAM (FA1/FA2; FA3 is Hopper-only, so skip it on a 3090), what a Gated DeltaNet layer's fixed-size state is, and why that makes 48 of 64 layers cache-free. This explains *why* the 64 KiB/token figure is what it is. | §I3 | M | P2 |
| **W12** | **"Where to learn this" at the point of need** | Each deep block ends with one line linking the matching Reference learning-path track or paper ("Want the full picture? Track 2, lectures 7-8" / "Paper: 2211.17192, with an agent prompt in Papers worth reading"). This bridges newcomers to the learning path instead of leaving it at the end. | §II4 | S | P2 |

## 4. Sequencing

1. **W2** (build support) and **W1** (start the rig harness) in parallel. W2 is local; W1 runs on the PC. Ollama/llama.cpp/vLLM installs on the rig are checked first, then the harness runs as a supervised job. The 15-minute alive check and Mac mini supervisor registration follow CLAUDE.md §4.
2. **W3, W4, W5:** the newcomer pass. These need no rig data and can ship as their own PR.
3. **W7, then W8, W6, W9, W10:** the technical blocks, each written around W1/W7 rows.
4. **W11, W12** last.

Proposed PRs: (A) W2 + W3-W5, the newcomer track; (B) W1 harness + `bench/` data; (C) W6-W10, the Under-the-hood blocks citing `bench/`; (D) W11-W12.

## 5. Acceptance criteria

- `node scripts/verify.mjs` passes. New checks: balanced deep blocks, a checkpoint in every Part 1 section, every internal anchor resolves, and new widget math against hand-computed values (roofline intensity, expected tokens per pass, block-quant MSE).
- **Newcomer:** from the top of the page, a reader with no terminal experience reaches a model's first answer by following only the "Start from zero" section, with no undefined term on that path. Test: read it with every deep block closed and list each term used before its definition; the target is zero.
- **Technical:**
  - At least 12 measured rows in `bench/` from the 2×3090 rig, each with build hash, driver, power cap and PCIe link.
  - Every "Not measured on a 3090 yet" / "Untested on a 3090 here" callout is either replaced by our row or kept with a reason (e.g. "-sm tensor failed on qwen35: log excerpt").
  - The guide shows one nsys timeline and one dmon trace from our rig.
  - Measured α for MTP on Qwen3.8-27B.
  - A measured KLD table for at least four quants.
- **Side by side with the frontier:** the roofline and all-reduce blocks state the same results as *How To Scale Your Model*'s inference chapter (same formulas, cited) and add our consumer-card measurements, which it lacks. "Start from zero" reaches a first answer in no more steps than Ollama's quickstart.
- No number without a label (measured here / primary / derived / secondary). Every `[unverified]` in research §I-§IV is resolved or still labelled when it ships.
- Rendered and checked: open and closed states in the default theme and one dark theme, 1400 px and 360 px widths, and no horizontal scroll.

## 6. Decisions for Joona (max 4)

1. **Deep blocks closed by default?** Recommended: yes, closed, with the rail toggle to open all. The alternative is a page-level "Beginner / Under the hood" switch that hides content. That is simpler for the reader, but hidden content can't be searched with Ctrl+F.
2. **No-terminal route:** recommend LM Studio as the zero-terminal first answer (it is free, but closed-source). Or keep Ollama only, which is open source and needs a terminal.
3. **Rig runs:** can the harness run on the 2×3090 box unattended (supervised job, logs to `bench/`), or do you want to start each run yourself?
4. **Publish our screenshots/traces from the rig?** They show hostnames and paths. Recommended: yes, after a scrub step in the harness that strips host, user and paths.

## 7. Top 5 proposals (summary)

1. **W1, measure the rig** (P0): turn every "ceiling / secondary / untested here" 3090 number into our own dated `bench/` row. This is the single biggest credibility and depth gain.
2. **W2, one stream, two depths** (P0): closed "Under the hood" blocks plus checkpoint callouts, enforced by `verify.mjs`.
3. **W3, Start from zero** (P0): terminal-free and terminal routes to a first answer with expected output, and the jargon-heavy path table moved after the plain explainer.
4. **W7 + W8, profile two GPUs** (P1): `dmon`, `nsys` and `nccl-tests` on our cards, showing why two 3090s give about 1.6× and not 2× (per-call all-reduce latency, not PCIe bandwidth).
5. **W9 + W10, the maths behind the speed-ups** (P1): speculative acceptance (expected tokens = (1−α^(γ+1))/(1−α)) and quantization error (Δ²/12, outliers), each with a widget and our measured α and KLD.

---

# Plan: Qwen3.8-27B on two RTX 3090s, quantization taught properly, larger models (2026-11-03)

> **Status:** IMPLEMENTED 2026-11-03 (ITER below, qwen-guide#16) after Joona's "implement it all", with the §5 recommended defaults: rig-free parts shipped, Flash-Next as one licence paragraph, quantization as its own section, Q4_K_M stays the headline with UD-Q4_K_XL as the context option. W1-W5, W7-W10 shipped, plus W6 text and the vLLM section (#15). Still open: §3 measurements on the 2×3090 rig (not built yet; the owner's PC has an RTX 5060 Ti 8 GB), which replace the derived ceilings.

## 1. Goal

Joona now runs two RTX 3090s (48 GB VRAM). The guide should make **Qwen3.8-27B** (5 Aug 2026, the newest 27B; no Qwen4 weights exist) the headline model for 24-48 GB readers. **Qwen3.5** stays the pick for 8-16 GB machines. Readers should come away understanding quantization: what the formats are, what the bits cost, and how to measure the loss. A new section covers models larger than 27B that 48 GB opens up. Acceptance criteria are in §4.

**Frontier reference:** Unsloth's Qwen3.8 docs page and the SGLang Qwen3.8-27B cookbook are the best current "run this model" pages. Neither covers Ampere or two consumer cards, and neither teaches quantization from first principles with measured KLD. Beating them means: a 2×3090 recipe that we have run ourselves, with measured t/s and KLD numbers from our own machine.

## 2. Changes, by section

| # | Section | Change | Evidence |
|---|---|---|---|
| W1 | Hero + "Pick your path" | The 24 GB and 48 GB paths name Qwen3.8-27B. The 8-16 GB path keeps Qwen3.5-4B/9B. Add "checked 3 Nov 2026". | §A |
| W2 | Which Qwen? | A "Qwen3.8-27B" subsection: what it is (dense, 27.78B, vision, Apache-2.0, 262K native), the 3.5 → 3.6 → 3.8 lineage with vendor benchmark deltas (labelled vendor-reported), and thinking defaults, `reasoning_effort`, `preserve_thinking` and both official sampling presets. The 3.5 table stays as is. Remove "on a 24 GB machine prefer 3.8" from the 3.5 row and move it into the new subsection. | §A |
| W3 | **New section "Quantization, from first principles"** (Part 1, after "Tokens, context and the KV cache") | What a weight is. Why bits = memory = speed (bandwidth ceiling). bpw, with the worked example 19.0 GB → 5.5 bpw. Block quantization (k-quants, the 2212.09720 block-size finding). imatrix and "dynamic" per-layer quants. GPTQ/AWQ (W4A16) against GGUF against EXL3 against FP8/NVFP4, and which GPU runs which. The research table (where the papers agree, where they conflict, what is untested). A **"bigger model at lower bits, or smaller model at higher bits?"** rule. How to measure the loss yourself: `llama-perplexity --kl-divergence` against a BF16 or Q8_0 baseline. KV-cache quantization moves here. | §F |
| W4 | Quant widget (new, `template/widgets/quant.html`) | Pick model (3.5-9B, 3.8-27B, 3.5-122B-A10B) and quant. Shows file size from the real HF table, bpw, the decode ceiling on 1 or 2 3090s, and the context that fits in 24/48 GB using the 64 KiB/token KV formula. The math is exposed on `window` and tested in `verify.mjs`, like FineTuneMath. | §B, §C, §D |
| W5 | Running on an NVIDIA PC | Rewrite the one-3090 table around Qwen3.8-27B quants (UD-Q4_K_XL 17.6, Q4_K_M 19.0, UD-Q5_K_XL 20.9, UD-Q6_K 22.0) with context-fit numbers. The KV note becomes "proven for 3.8" and drops UNCERTAIN. | §B, §C |
| W6 | **Two RTX 3090s, rewritten as a full path** | Three recipes, each run on Joona's rig before publishing. (a) llama.cpp layer split, Q8_0, 128K context. (b) llama.cpp `-sm tensor` (experimental), with a stated result even if it fails on qwen35. (c) vLLM TP=2 with AWQ-INT4. Speculative decoding with the MTP/DFlash sidecars (`--spec-type draft-mtp`). Measured tg/pp at 0, 32K and 128K context. Power cap results. | §D, §G |
| W7 | **New section "Bigger than 27B"** | Qwen3.6-35B-A3B Q8. Qwen3.5-122B-A10B at IQ3_XXS fully on GPU against Q4 with `--n-cpu-moe` and `--moe-cache-mib`. A 70B dense for comparison. Qwen3.8-Flash-Next (~180B) as "needs ≥96-128 GB RAM", with its **Qwen Community License** conditions spelled out. Dense against MoE, and why active parameters set speed. | §E |
| W8 | Faster and smaller | Fix the conflict: the "quantized models think longer / truncation doubles" claim is downgraded to one report and set against 2504.04823. Point readers to W3 for the theory. | §F |
| W9 | Fine-tune Part 3 | Add a 2×3090 row for a 27B LoRA, using Unsloth's existing figures (bf16 LoRA 27B 56 GB, so it does not fit in 48 GB; QLoRA ~22 GB fits on one card, though Unsloth does not recommend QLoRA on the 3.5 architecture). Keep it short; no new training recipe. | old research §5 |
| W10 | Learn more | Add the quantization papers (GPTQ, AWQ, 2212.09720, 2504.04823) and llama.cpp's quantize/perplexity docs. | §F |

## 3. Measurement plan on the 2×3090 rig (before any number ships)

All runs use `llama-bench` / `llama-perplexity` from llama.cpp ≥ v0.8.0 and vLLM latest stable, with one variable at a time, and are logged to `bench/` in the repo:
1. tg128 / pp512 for Qwen3.8-27B at UD-Q4_K_XL, Q6_K and Q8_0, with `-sm layer` and `-sm tensor`, at context depth 0, 32K and 128K.
2. MTP and DFlash speculative decoding, on and off (`--spec-type draft-mtp` / `draft-dflash`), on 20 fixed prompts: tok/s and acceptance rate.
3. KLD and perplexity of each quant against Q8_0 (BF16 is 53.8 GB, which won't fit on 48 GB; say so) on wikitext-2 plus a code sample.
4. vLLM TP=2 AWQ-INT4: single-user tok/s and 4-way concurrent throughput.
5. Qwen3.5-122B-A10B IQ3_XXS fully on GPU against Q4_K_M with `--n-cpu-moe` (depends on the rig's RAM).
6. Power cap at 280 W against stock on recipe 1.

## 4. Acceptance criteria

- `node scripts/verify.mjs` passes, with new tests for the quant widget math: file-size lookup, bpw, ceiling, and KV fit against hand-computed values.
- Every number in the new text traces to `research.md` §A-G or `bench/`. Secondary sources are labelled. No UNCERTAIN item is stated as fact.
- At least one 2×3090 recipe has been run end to end on the rig, with a measured tok/s.
- Browser check: 320 px, all 9 themes, no overflow. The new widget works by keyboard.
- The 3.5 content for 8-16 GB machines is still there and unchanged except W8.

## 5. Decisions for Joona

1. **Where are the 3090s?** The current PC reports one RTX 5060 Ti 8 GB and 31 GB RAM today. Is the 2×3090 build a new machine, or an upgrade to the current PC, and how much system RAM will it have? This sets W6/W7's measured claims. 64 GB+ unlocks the 122B MoE at Q4. **Recommended:** publish W1-W5 and W8-W10 now (they need no rig); W6/W7 ship after §3 runs.
2. **Flash-Next in the guide?** It is not Apache (Qwen Community License 1.0) and needs ≥96 GB RAM. **Recommended:** one paragraph with the licence summary, no recipe.
3. **Quantization as its own section, or folded into "Faster and smaller"?** **Recommended:** its own section (W3). It's the teaching core you asked for; "Faster and smaller" keeps the engines.
4. **Lead model for 24 GB single-card readers:** Qwen3.8-27B at UD-Q4_K_XL (17.6 GB, more room for context), or Q4_K_M (19.0 GB)? **Recommended:** UD-Q4_K_XL, pending the KLD run in §3.3.

---

# Plan: major improvements to the Qwen guide

> **Status:** IMPLEMENTED 2026-10-16 (ITER-008), after Joona said "implement it all" with the recommended defaults for the four decisions in §5. Evidence for every item is in `research.md`.

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

Implemented 2026-10-16 with the recommended defaults (see ITER-008 in BUGS_AND_ITERATIONS.md).
