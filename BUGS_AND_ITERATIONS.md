# Bugs & Iterations — Qwen, locally

## 2026-10-25: ITER: "Candy" theme renamed "Bubblegum"

- **What:** Suite-wide rename (Joona). The theme key `candy` → `bubblegum` in `template/style.css`, `template/base.js`, `scripts/build.py` and the README, and `index.html` is rebuilt. A reader whose saved `qft-theme` is `candy` is moved to `bubblegum` on load (`apply()` re-saves it).
- **Verify:** `python3 scripts/build.py --check` passes. In local Chromium, a stored `candy` loads as `data-theme="bubblegum"`, storage is rewritten, the menu shows "Bubblegum" checked, the background is `#fbf3f5`, and there are no console errors.

Newest first.

## ITER-008 — v2: RTX 3090 / NVIDIA PCs, depth, and a free learning path (2026-10-16)
- **Why:** owner feedback: the guide still read as Mac-only ("3090" appeared once, "VRAM" never), wasn't robust
  enough, and pointed to almost no free learning resources (Karpathy appeared once). An 11-agent audit found 22 gaps
  (5 P0); `plan.md` set out 12 work packages and `research.md` the verified evidence (79 verified, 8 refuted, 31
  downgraded). Implemented with the owner's four defaults: learning path in Reference, 3090 speeds as cited
  estimates, Qwen3-Embedding-0.6B at 1024 dims end to end, AMD/Intel/CPU as a router row only.
- **New sections:** Pick your path (hardware + goal router in the intro); Running on an NVIDIA PC (cards table with the
  llama.cpp #15013 scoreboard, what fits on one 3090, decode ceilings, prefill vs decode, driver floor, engine ladder,
  power limiting, two 3090s, smaller cards, buying used); Tokens, context and the KV cache (formula, chat templates,
  quant formats by GPU generation); Benchmark your machine; Route the work (split from Harden); A free learning path
  (tracks 0–6 + stay current, Karpathy's Zero to Hero at the centre); Glossary.
- **Expanded:** Install (Mac / Linux systemd / Windows settings, WSL2 rules, 24 GB NVIDIA set), Other runtimes (PC
  table), Thinking mode (+ sampling from the official card), Use it from code (+ structured output, tool calling,
  images, compare-models harness, speculative decoding on a GPU), Part 2 (one embedder end to end, single-24-GB-PC
  budget, `--gpus all`, NVIDIA build-order alternatives, Learn-more links), Part 3 (5090 / 2×24 GB / 27B rows,
  per-card script comments), Troubleshooting (Area column, 12 new rows), Sources (hardware and PC runtimes).
- **New widget:** `fit.html`, a fit and decode-ceiling calculator for 11 NVIDIA and Mac presets; decode widget gains
  RTX 3090 and 5090 presets.
- **Process:** 9 writers returned edits as data (59 applied, 0 anchor failures); 4 skeptics then checked 198 claims and
  found 7 problems. 6 were fixed: the Linux/Windows model paths, the dual-3090 memory rule, the Vulkan/AMD wording, and
  Ollama's default context (it is VRAM-dependent, 32K on a 24 GB card, per docs.ollama.com/context-length; the FAQ's
  4096 is stale). A wrong cross-reference was also fixed. One was skipped: a list-indent change that suits CommonMark
  but would break this guide's own renderer.
- **BUG-006:** the fit calculator called Gemma 4 31B "Fits" on a 3090 although the guide calls it the tightest fit.
  Root cause: with no verified KV constant, the KV cache counted as zero. Fix: an unknown KV with under 5 GB of
  headroom is "Tight", with a note; two tests added.
- **BUG-007:** a generic `C:\Users\%username%` path tripped the home-path guard; replaced with `%USERPROFILE%`.
- **Verified:** `verify.mjs` PASS (310 checks, up from 223); all 150 URLs load (localhost samples excepted; the
  OWASP page blocks scripts but loads in a browser); 48 sections, no broken in-page anchors, no console errors, no
  horizontal overflow at 1400px or 360px.

## ITER-007 — Absorb Qwen Local into this guide; four parts (2026-10-15)
- **Why:** two public Qwen guides overlapped (running, fine-tuning and abliteration were in both) and linked each
  other. Joona asked to fold the older `qwen-local-guide` into this one and delete it.
- **New Part 2, "Build on it":** embeddings and rerankers, retrieval (RAG) and the upgrades past top-k, a Qdrant
  vector store, shared memory over MCP, the security record and hardening, routing work across a local box, a GPU
  machine and a hosted model, a week-by-week build order, and a Learn-more list. Fine-tune is now Part 3 and
  Abliteration Part 4; every cross-reference, the hero, nav, meta description and README follow.
- **New Part 1 section, "Faster and smaller":** MLX versus llama.cpp, speculative decoding, and quantization past
  Q4_K_M (Unsloth Dynamic, IQ4_XS, DWQ, the thinking-mode quality cliff, asymmetric KV cache, QAT and ternary).
- **Calculators ported:** decode-speed ceiling (Part 1), 16 GB stack budget and compounding reliability (Part 2).
  Math on `FineTuneMath`; `verify.mjs` asserts the doc's own figures (120 ÷ 6.6 ≈ 18, 0.95⁸ ≈ 66%, budget bands).
- **How it was written:** a workflow drafted each section from the old source with a fidelity skeptic and a reviser
  per section (every number traced to the old guide; personal context such as named machines and agents removed and
  generalised). A second check found 4 gaps, all fixed: Ollama's MLX default qualifies the engine verdict, a lost
  "about", the "BFCL v4 rated Flawed" caveat, and the develop-locally/serve-on-GPU pattern.
- **BUG-006:** the budget checkboxes split into columns at 360px because `.check{display:flex}` treated the inline
  `<code>` as a flex item. Fix: wrap the label text in a `<span>`.
- **Verified:** `verify.mjs` PASS (223 checks); 1400px and 360px with no horizontal overflow and no console errors.
- **Old repo:** `Joona-t/qwen-local-guide` backed up as a git bundle of all branches outside the working tree, then
  deleted on request.

## ITER-006 — "A program, written as weights" widget in Part 2 (2026-10-15)
- **Why:** Part 2 says weights are numbers and that training changes them, but never showed *why* numbers can hold
  behaviour. Cheng Lou's [handwired](https://github.com/chenglou/handwired) (MIT) rewrites ordinary JS control flow
  as multiplies, adds and ReLU; its first example is the whole argument in two lines.
- **What:** `template/widgets/handwired.html` + a `calc-wired` block in `template/widgets.js`: the program
  `a > 4 ? 10 : 12` beside its rewrite `above4 = relu(a − 4) − relu(a − 5); result = above4·10 + (1 − above4)·12`,
  with a slider for `a` (0 to 10, step 0.25) and both results live. The math is exposed on `FineTuneMath`
  (`relu`, `wiredProgram`, `wiredGate`, `wiredNetwork`). Placed after the "four ways to change behaviour" table with
  a lead-in that credits the repo and links the live demo. Nothing from the repo is loaded; the page stays one file.
- **Honest edge:** the rewrite is exact for whole numbers only. Between 4 and 5 the gate ramps, so `a = 4.5` gives 11
  where the program gives 10. The widget's note and caveat say so rather than hiding the slider's fractional stops.
- **Tests:** `verify.mjs` asserts agreement for every whole `a` in 0..10 (plus −3, 3.9, 5.1 and 1e6), the gate stays
  in [0, 1], and the ramp values at 4.25 / 4.5 / 4.75 (191 checks, PASS).
- **Verified:** 1400px two columns side by side, 360px stacked, no horizontal overflow (scrollWidth = 360), no
  console errors; Candy and Noir checked.
- **Review (3 auditors + 6 skeptics, all 6 findings confirmed, 0 debunked) fixed before commit:** caveat overclaimed
  "whole numbers only" (the rewrite is exact everywhere outside the 4-to-5 ramp); `.metrics--1` lost the cascade to a
  later `.metrics` rule; the live region wrapped five bare numbers while the summary sentence sat outside it (now the
  `.calc__out` pattern the other widgets use); `<p class="wired__tag">` picked up the 18px body-text rule; two
  identical formatters.

## ITER-005 — "Expand table" control on wide tables (2026-10-15)
- **What:** tables with five or more columns get an Expand/Collapse button. Expanded, the table breaks out of the
  700px reading column: rightwards to the page gutter when the contents rail is visible, centred to the full width
  when it isn't. The button is hidden under 1080px, where the table already spans the column and scrolls.
- **BUG-005:** the first CSS-only version overlapped the sticky contents rail and ran 59px past the right edge
  (measured). Fixed by sizing in script from the column's measured left edge, re-run on resize.
- **Verified:** expanded table clears the rail and stays inside the gutter at 1400px; collapse restores the column
  width; `verify.mjs` PASS.

## ITER-004 — Landscape rebuilt from an adversarially verified Oct 2026 survey (2026-10-15)
- **Why:** the owner flagged that "The local landscape" omitted the current frontier (GLM-5.3, Kimi K3) and carried
  stale facts, which undermines the guide as a source. A 12-agent workflow (5 web scouts by lab group, 1 stale-claim
  auditor over the whole source, 5 adversarial skeptics, 1 synthesizer; 130 claims verified, 33 refuted, 53
  downgraded) surveyed the open-weight landscape; the ten load-bearing surprises were then re-checked by hand on
  primary pages (Google's Gemma 4 model card, Meta's Muse Glimmer card, Apple's Mac mini specs, Liquid's LFM2.5 card,
  DeepSeek-V4.1-Flash and GLM-5.3 and Kimi K3 cards, Mistral's Large 4 post, Ollama tag pages).
- **BUG-002 — Gemma 4 license was wrong.** The table said "Gemma terms of use"; Google's model card says Apache 2.0.
- **BUG-003 — Llama row was obsolete.** Meta's current open model is Muse Glimmer 30B (Aug 2026, Apache-2.0); no Llama
  has shipped since Llama 4 (Apr 2025).
- **BUG-004 — Bandwidth example named the wrong current Mac.** The base Mac mini is now M6 at 153 GB/s (170 at 24/32
  GB); the M4 figure is kept as the 2024 example.
- **Landscape section rewritten as two tables:** "Fits a laptop" (14 rows, 4-bit sizes from Ollama/official GGUFs)
  and "The open-weight frontier" (12 rows incl. Kimi K3 2.8T, Qwen3.8-2.4T, DeepSeek-V4.1-Flash, GLM-5.3 and
  5.3-Flash, Mistral Large 4 as announced-not-open, MiniMax-M3, gpt-oss-120b, Llama 4, Nemotron 3 Ultra, Mistral
  Small 4, Qwen3-Coder-Next), plus "why the frontier matters" and a dated watch list of unverified items.
- **Other fixes:** Qwen3.8-27B named as the 24 GB pick; Qwen3.8 line licenses spelled out; Ollama MLX timeline
  (v0.19 preview → v0.40.0 default, no memory floor in the release notes); thinking-default caveat (chat templates vs
  Unsloth's page disagree); mlx-lm `--adapter-path` regression cited to issue #1248; OBLITERATUS wording softened;
  caveats now date the snapshot.
- **Not adopted from the synthesis:** its proposal to say all Qwen3.5 sizes are non-thinking by default (the chat
  templates, verified earlier, say 4B/9B think by default); its "no evidence" verdict on the mlx-lm regression (the
  issue exists). Secondary-only claims (Qwen4 "in training", Nemotron 3.5 Bolt, Mellum2.1 benchmarks) went to the
  watch list, not the tables.
- **Verified:** `verify.mjs` PASS (125 checks); tables render and scroll inside their wrapper at 360px.

## ITER-003 — One guide in three parts: run, fine-tune, abliteration (2026-10-15)
- **Why:** the owner wanted one complete local-Qwen guide, not a fine-tuning guide with pointers elsewhere: set up
  first, then fine-tune, then abliteration, each with learning resources, plus responsibility and safety; and a
  section on Hugging Face; and a view of local models as a whole so readers choose by use case, not by hype.
- **Part 1 (new):** what a local model is; "local models and your use case" (when local wins, when a hosted model is
  still right, a landscape table of Qwen/Gemma/gpt-oss/Ministral/Granite/Llama/Qwen2.5-Coder with sizes and
  licenses, a six-step how-to-choose, benchmarks-are-a-starting-point); which Qwen3.5 size; Hugging Face (the Hub,
  model cards, safetensors/GGUF/MLX, the `hf` CLI from its docs, tokens, running Hub GGUFs in Ollama with the
  `hf.co/user/repo:QUANT` syntax from HF's Ollama page, licenses); Ollama install on macOS/Linux/Windows with GPU
  requirements from docs.ollama.com/gpu; thinking mode; other runtimes; the OpenAI-compatible API; keeping it on
  loopback; learn-more list. Setup facts for the Mac reuse the skeptic-verified Qwen Local research.
- **Part 2:** the existing fine-tuning sections, renumbered, with a learn-more list.
- **Part 3 (new):** abliteration mechanism, later research (five papers), measured costs, a full responsibility and
  safety section (what is removed, who you are responsible to, five rules, a five-question decision check, why the
  section explains rather than refuses), three routes, evaluation, learn-more list including the interpretability
  background. Conceptual and tooling level only; no prompts, no editing code.
- **Build:** `<!-- part title="…" -->` markers group the contents rail; hero, nav and rules rewritten; source renamed
  to `source/qwen-guide.md`; troubleshooting table gains Part 1 and Part 3 rows; sources and caveats extended.
- **Verified:** `verify.mjs` PASS (125 checks, 34 sections); every learn-more link returned 200 on 2026-10-15
  (two candidates that didn't were dropped); browser check at desktop and 360px.

## ITER-002 — Editorial redesign, closer to the Raw Thought reader (2026-10-15)
- **Why:** owner feedback. The candy "glass-sticker" shell read as too zoomed-in and childish for a general audience:
  big rounded cards, letter-spaced OpenDyslexic headings, and a busy pill row.
- **What:** a new `template/style.css` replaces the inherited shell CSS. It follows the Raw Thought reader's layout:
  - a hairline top bar with an italic serif wordmark;
  - a hero with an eyebrow, a large serif title, an italic date line, and the three rules in a double-ruled plaque;
  - a 700px serif reading column with a slim sticky contents rail that highlights the current section, collapsing
    to a `<details>` list under 1080px;
  - hairline tables, ruled callouts with plain-text labels, flat code blocks, and widgets as quiet double-ruled
    panels;
  - a reading progress bar.
- **Kept:** the four themes, re-tuned as flat paper/ink/accent palettes, with every text pair at AA or better
  (≥ 4.8:1, computed). Sparky stays in the wordmark and footer.
- **Added:** a reading-font choice (Serif, Sans, OpenDyslexic) in the Change Theme menu, persisted per browser.
- **Removed:** emoji from widget titles and callout labels.
- **Verified:** `verify.mjs` PASS (87 checks); no console errors; 1400px, 800px and 360px with no horizontal
  overflow; Candy/serif and Noir/OpenDyslexic screenshots checked.

### BUG-001 — "Custom (billions)" field showed for preset models (2026-10-15)
- **Symptom:** at 360px in the Candy theme, the memory calculator showed the custom-parameters input even with
  Qwen3.5-4B selected.
- **Root cause:** the shared `.field{display:flex}` rule beats the browser's default `[hidden]{display:none}`.
- **Fix:** `.field[hidden]{display:none}` in `template/widgets.css`. Verified in the browser: `none` for a preset,
  `flex` for Custom.

## ITER-001 — First public-ready version (2026-10-15)
- **Why:** the goal is a fine-tuning guide anyone can use. It grew out of an earlier private planning repo that was
  tied to one personal workflow. Rather than publish that repo and its history, this one starts clean and keeps only
  the general teaching parts.
- **Reused from the earlier plan:** the single-weight gradient demo, the LoRA parameter counter, weight-storage
  arithmetic and the quiz idea, with the same math and its tests ported to `scripts/verify.mjs`. The examples were
  rewritten to be general.
- **New content:** prompt vs RAG vs tools vs fine-tune; a hardware fits table for Macs and NVIDIA cards; a
  decide-first checklist (baseline, held-out test, pass rule, stop rule); a dataset checklist with thinking-mode
  handling; Path A (`mlx-lm` 0.32.0); Path B (Unsloth on Linux, WSL or Windows, GGUF → Ollama); evaluation,
  shipping, DPO/GRPO/vision pointers; troubleshooting; sources; caveats.
- **Evidence:** Mac facts reuse the skeptic-verified research behind the Qwen Local guide's Stage 7 (2026-10-15).
  NVIDIA facts were fetched from Unsloth's Qwen3.5, install, Windows and Ollama pages that day: bf16 LoRA VRAM table,
  QLoRA-not-recommended quote, transformers v5, the uv/conda install commands, `save_pretrained_gguf`, and Modelfile
  auto-generation. The training script's model, LoRA and trainer arguments are Unsloth's; the dataset-mapping lines
  are this guide's and are labelled as such. Secondary and untested items are labelled inline.
- **Build:** the Markdown is the source of truth and `scripts/build.py` renders `index.html`, so the two can't drift
  (the sibling guide's BUG-013). `scripts/verify.mjs` fails on a stale build, wrong widget math, broken anchors,
  network calls, or personal paths and names.
