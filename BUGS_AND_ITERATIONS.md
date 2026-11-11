# Bugs & Iterations — Qwen, locally

## 2026-11-11: ITER: Linux for local AI, and vLLM vs Ollama rebuilt around two GPUs

**Why:** Joona asked for the guide to push Linux, and to make vLLM the clear pick for more than one consumer GPU. He runs Ubuntu on his two-RTX-3090 PC with a small Windows install for Windows-only work. He found a fresh Windows install more frustrating than Ubuntu, and Linux easier to work in with agents. He also pasted a common "Ollama for development, vLLM for production" comparison to check.
**Finding:** the old section said "with one user, the gap mostly disappears". That holds on one GPU, but not on two. Ollama's FAQ: a model that fits on one GPU is loaded on that GPU, so a 19 GB Qwen3.8-27B leaves the second 3090 idle; a bigger model is spread by layers and the cards take turns. A dual-3090 test from October 2026, with no NVLink (computingforgeeks, one rig, secondary), measured tensor parallelism at 1.5 to 1.7 times one card for a single user. On a 70B, layer split gave 19.5 t/s against 31.9 for llama.cpp `-sm tensor` and 31.3 for vLLM. The pasted table's 41 vs 793 t/s and 673 vs 80 ms are Red Hat's single-A100, 256-user peak test. The real deciding factors are concurrency and the number of GPUs, not development against production.
**Change:**
- New section "Linux: the platform local AI is built on" (#linux), between Install and the NVIDIA PC section. It covers:
  - why Linux;
  - the author's note;
  - Ubuntu LTS against Omarchy (DHH, Arch + Hyprland, v4.5.0; "we haven't used it");
  - the dual-boot clock fix (`timedatectl set-local-rtc 1` with no `--adjust-system-clock`);
  - Ubuntu NVIDIA driver steps from Ubuntu's docs;
  - the commands you'll use;
  - free Linux courses (Canonical tutorial, MIT Missing Semester 2026, Linux Foundation LFS101 at $0 and 60 h, OverTheWire Bandit, The Linux Command Line, ArchWiki NVIDIA).
- vLLM section retitled "vLLM vs Ollama: two GPUs, many requests, and what each one is", with:
  - a layers table (weights, engine, server, API, front-end, harness) explaining what Ollama and vLLM are;
  - "What changes with two GPUs", with the measured table;
  - an NVLink callout: the docs' pipeline-parallel advice is about throughput, not single-user speed;
  - a note on the pasted comparison table;
  - "one person, one GPU" and two new two-GPU rows in "Which to use".
- vLLM install updated to the docs' uv steps (`uv venv --python 3.12 --seed --managed-python`, `uv pip install vllm --torch-backend=auto`, `vllm download-kernels`), v0.33.0 (2026-11-02).
- Router: the two-3090 row now says "Linux, then llama.cpp `-sm tensor` or vLLM"; the 24 GB row says Linux is recommended.
- The Windows/WSL paragraph now recommends Linux.
- The two-3090 split modes gain the measured `-sm tensor` result and the `-sm row` CUDA failure.
- The learning path points to the Linux courses.

**Verified:** `node scripts/verify.mjs` PASS (389 checks); Omarchy, Ubuntu driver, vLLM install, Ollama FAQ, LFS101 and Missing Semester pages fetched on 2026-11-11; all new links return 200; renders at 1400 px and 360 px with no overflow and no console errors.

## 2026-11-05: ITER: "Papers worth reading" section, plus prompts for agents

- **What:** Joona asked for recommended papers, and for suggesting that readers have their agents read them or search for recent papers and look for abstractions to reuse. New Reference section `#papers`: how to read a paper (with two paper-and-rebuttal pairs), a 35-entry list by topic with the 2026 preprints marked, two copyable agent prompts (read one paper; search the last 90 days), the `pdftotext` route for small context windows, warnings that papers are untrusted input and that agents can invent citations, and a one-change-at-a-time closing. Linked from the intro, the goal table and "Stay current".
- **Bug caught before merge:** three details written from memory were wrong: an arXiv ID (2306.12001 is not Representation Engineering) and the titles of 2504.04823 and 2306.00978. **Root cause:** citing from memory. **Prevention:** `scripts/check_papers.py` compares every link in the section with its arXiv title; link text must be the exact title. It needs network, so it's a manual step, not part of `verify.mjs`. Enforcement is therefore partial.
- **Verify:** `node scripts/verify.mjs`: PASS, 384 checks. `check_papers.py`: 35 papers PASS, and FAIL on a deliberately wrong ID. Both `curl` commands and the `pdftotext` steps were run; the Minions paper came to 18,179 words. Headless Chrome at 1400 px and 390 px. Notes come from abstracts, and the section says so. **Not tested:** neither prompt has been run on a local model (no Ollama on this Mac), and the section labels them untested.

## 2026-11-04: ITER: Basalt is the default theme again

- **What:** Joona reversed the Holo Mochi default the same day: Basalt is "more long read friendly". Reverted `scripts/build.py` and the `template/base.js` fallback to `basalt`. Readers whose browser saved `kr-holo` during the few hours it was the default keep it (saved choices win).
- **Verify:** `python3 scripts/build.py` then `node scripts/verify.mjs`: PASS, 367 checks. Local preview with storage cleared: `data-theme="basalt"`, `data-font="serif"`, Basalt menu item `aria-checked="true"`; hero screenshot shows Basalt in Cinzel.

## 2026-11-04: ITER: Holo Mochi is the default theme

- **What:** Joona asked for Holo Mochi (Korean set, `kr-holo`) as the default. Changed the `<html data-theme>` in `scripts/build.py` and the no-saved-choice fallback in `template/base.js`. Saved choices still win, so readers who already have a theme stored keep it.
- **Verify:** `python3 scripts/build.py` then `node scripts/verify.mjs`: PASS, 367 checks. Local preview with storage cleared: `data-theme="kr-holo"`, `data-font="serif"`, Holo Mochi menu item `aria-checked="true"`; hero screenshot shows the Holo Mochi palette.

## 2026-11-03: ITER: LLM101n callout links the repo and says what it is

- **What:** Joona pointed at github.com/karpathy/LLM101n. The old callout ("LLM101n is not a course… has not been released") was correct, but it gave no link and no reason. It now links the repo and quotes the README ("this course does not yet exist", being built by Eureka Labs). It also gives the archive date (2024-08-01, only a README and an image) and maps the syllabus onto the free materials that already exist (Zero to Hero, minbpe, build-nanogpt, nanochat).
- **Checked 2026-11-03:**
  - GitHub API: archived, last push 2024-08-01, contents README.md and llm101n.jpg.
  - eurekalabs.ai: "heads down building LLM101n" (2024-07-16), with no release.
- **Verify:**
  - `node scripts/verify.mjs`: PASS.
  - `freshness.py`: PASS.

## 2026-11-03: ITER: Qwen3.8-27B plan implemented (quantization section, quant calculator, bigger models)

- **What:** Joona said "implement it all". The 2026-11-03 plan is in, with the recommended §5 defaults.
  - **W2:** a "Qwen3.8-27B, the 24-48 GB pick" subsection: lineage, vendor benchmarks labelled as such, and which build for which memory.
  - **W3:** a new section, "Quantization, from first principles":
    - why bits are memory and memory is speed, with a worked table (BF16 to UD-Q3_K_XL);
    - why "4-bit" is 5.5 bpw (Q4_K is 4.5 bpw per `ggml-common.h`; Q4_K_M bumps attn_v/ffn_down per `llama-quant.cpp`);
    - the research table (2212.09720, 2504.04823, 2404.14047, 2502.13178): where the papers agree and what is untested;
    - the bigger-vs-higher-bits rule;
    - measuring KLD with `llama-perplexity --kl-divergence-base/--kl-divergence`;
    - the formats table, moved here.
  - **W4:** a quant calculator (`template/widgets/quant.html`). It uses real GGUF sizes for the 9B, the 27B and the 122B-A10B, with params from the safetensors totals and KV KiB/token from config (32/64/24). It shows bpw, the decode ceiling, and context at f16/q8_0. The maths is in `FineTuneMath.quantCalc` with 16 new verify tests.
  - **Fit widget:** the Qwen3.8-27B KV is now verified (64 KiB), and UD-Q4_K_XL was added.
  - **W5:** one-3090 rows for UD-Q4_K_XL and UD-Q5_K_XL.
  - **W6:** MTP/DFlash and vLLM pointers for two cards, plus a "not measured on a 3090 yet" note.
  - **W7:** "Bigger than 27B", covering dense vs MoE, the 35B-A3B, 122B-A10B (IQ3_XXS just fits; Q4 needs RAM offload), 70B, and Flash-Next with its licence terms.
  - **W10:** quantization and vLLM learn-more links.
- **Caught in testing:** the calculator counted K as 1,024 tokens while the prose used 1,000 (60K vs 64K for the same 4 GB). Both now use thousands, and the one-3090 Q4_K_M row reads ~61K/~122K. The UD-Q4_K_XL row first said 88K; corrected to 82K.
- **Verify:**
  - `node scripts/verify.mjs`: PASS (367 checks).
  - Mutant: setting the 27B's KV to 60 KiB fails "Qwen3.8-27B constants".
  - `freshness.py`: PASS.
  - Browser at 320 px: 9 themes, no overflow; the calculator gives 61K/122K on a 3090 at Q4_K_M, 256K-capped on two 3090s at Q8_0, and "doesn't fit" for 122B Q8_0; keyboard focus reaches the selects.

## 2026-11-03: ITER: vLLM section, and why it beats Ollama for serving

- **What:** Joona asked for vLLM and why it is better than Ollama. The new Part 1 section "vLLM vs Ollama" covers:
  - PagedAttention (arXiv 2309.06180, 2-4×), continuous batching, prefix caching (on by default in v0.32.0) and tensor parallelism. Ollama spreads layers instead.
  - Red Hat's A100 benchmark: 793 vs 41 TPS, labelled as a vendor source.
  - The honest limit: with one user the gap mostly disappears, because decode is bandwidth-bound.
  - A which-to-use table.
  - What vLLM costs you: Linux/WSL2 only, compute capability ≥ 7.5, 92% VRAM by default.
  - A Qwen3.8-27B two-3090 recipe (AWQ-INT4 or FP8 as W8A16 via Marlin, TP=2, reasoning and tool parsers, MTP), labelled untested on a 3090.
- **Also:** the PC runtimes table now marks vLLM's GGUF support as an experimental plugin, and the two-3090 path row links the new section.
- **Evidence:** `research.md` §H. Every flag and default was checked against the v0.32.0 source or docs, and the recipe against the vLLM Qwen3.8 recipe.
- **Verify:**
  - `node scripts/verify.mjs`: PASS (325 checks).
  - `freshness.py`: PASS.
  - Browser at 320 px: section, table and code block render; 4 themes, no overflow.

## 2026-11-03: BUG: Stale scope and settled facts across the guide; freshness check added

- **Root cause:** the guide grew from a Qwen3.5 / 16 GB Mac guide into one covering Qwen3.6/3.8-27B, NVIDIA and two GPUs. New sections were added, but the framing text and the facts they settled were never revisited. Nothing failed when a "verified for 3.6 only" or "October 2026" line went stale. Joona caught the same class twice (the hero, then the opening callout).
- **Found by:** four independent Haiku sweeps of the whole guide (about 145 findings, about 45 unique), each re-checked against primary sources on 2026-11-03.
- **Fixed:**
  - Opening callout: now lists every tier it covers.
  - "Which Qwen?" framing: by memory tier.
  - KV-cache UNCERTAIN labels: removed in 7 places. The Qwen3.8-27B config is identical to 3.6's.
  - "Quantized models think longer": in 3 places, replaced with arXiv 2504.04823's result.
  - Qwen3.8-27B sampling presets and the `reasoning_effort` / `preserve_thinking` switches: added.
  - `-sm tensor` and `--spec-type draft-mtp/draft-dflash` (llama.cpp v0.8.0): added.
  - Context budget: L1321 (32K) now agrees with the fit table (about 48K f16 on 3 GB).
  - 24 GB Mac vs 18 GB model contradiction: fixed.
  - Muse Glimmer ceiling: 55 → 52 tok/s.
  - Watch list: Qwen4, Nemotron 3.5 Bolt and Mellum2.1, re-checked.
  - Mistral Large 4: now open.
  - Versions: Ollama v0.40.14, Qdrant v1.20.0, mcp-memory-service v12.5.0, mlx-lm 0.32.2, Heretic 2.0.0 (breaks 1.x configs).
  - huihui pulls: 717K.
  - Thinking defaults: verified from the Qwen3.5 chat templates.
  - `og:url`: pointed at a 404, now `/qwen-guide/`.
  - Wordmark aria-label: fixed.
  - Wordmark date: Nov 2026.
- **Enforced by:** `scripts/freshness.py`, run inside `verify.mjs`.
  - It fails on 8 stale patterns.
  - It fails on any "checked / as of / current on" date older than 45 days.
  - It reports the age of every dated claim.
- **Proven:** against the previous text (e6bb0e3) it reports FAIL (10), exit 1. On this text it passes.
- **Measured** (`/rsi`), before → after on the same instrument: stale-pattern hits 10 → 0; dated claims tracked 16 → 17; median age 18 → 0 days.
- **Verify:**
  - `node scripts/verify.mjs`: PASS (321 checks).
  - Browser at 320 px: 9 themes, no overflow; new sampling table renders.

## 2026-11-03: BUG: Hero said the guide was about Qwen3.5 only

- **Root cause:** the hero line was written before the NVIDIA sections added Qwen3.8-27B for 24 GB and two-GPU machines, and nobody updated it. Joona spotted it.
- **Fix:** it now reads "from Qwen3.5 on a laptop to Qwen3.8-27B on one or two 24 GB GPUs".
- **Verify:** `node scripts/verify.mjs`: PASS (320 checks).

## 2026-10-31: ITER: Korean themes from the papers archive

- **What:** At Joona's request the four Korean themes from lovespark.love/papers (Seoul Y2K 서울, Gingham Sticker 깅엄, Holo Mochi 모찌, Neon Usagi 네온) are in the **Change Font & Theme** menu under a "Korean" heading. `scripts/build.py` carries the papers archive's `KOREAN` table and its 4.8:1 contrast gate (every text colour against every background, card fill and gradient stop); the papers-only card selectors are mapped to this guide's `.calc` widgets and `.wbtn` buttons. The menu also takes the papers fix for Hangul breaking mid-word and for running off short screens (272 px, no-wrap, `keep-all`, capped and scrollable).
- **Verify:** `node scripts/verify.mjs`: PASS (320 checks); the build's contrast gate passes for all four. In the browser: each theme applies its palette, background and card radius; at 320 px, 4 themes × 3 fonts show no horizontal overflow and the menu scrolls inside the window.

## 2026-10-27: ITER: Basalt body in Cormorant Garamond

- **What:** Joona picked Cormorant Garamond, the face of lovespark.love's lore lines, as Basalt's reading font ("elegant"), upright after comparing upright and italic previews. Basalt's default Serif setting now sets body text in Cormorant Garamond 500 at 20.5 px / 1.6 (its x-height is small), bold at 600; headings stay Cinzel, labels DM Mono. Four latin woff2 faces (500/600, upright/italic, 86 KB, SIL OFL 1.1, from @fontsource/cormorant-garamond 5.2.5 on jsDelivr, downloaded with Joona's OK) are inlined in `template/fonts.css`, so the page stays offline.
- **Verify:** `node scripts/verify.mjs`: PASS. Headless Chrome at 1400 px: body renders in Cormorant Garamond with true bold and italic.

## 2026-10-27: ITER: Basalt wordmark shows the ankh; top nav folds at 1080 px

- **What:** At Joona's request Basalt shows lovespark.love's glyph ankh (`hero-ankh-beige.png`, resized to 160 px and inlined as `template/ankh.datauri`, 30 KB) in place of the wordmark mascot, with an ember glow. Other themes keep Sparky; Sparky stays in the footer in every theme.
- **Bug:** the longer "Change Font & Theme" label crowded the top nav between 860 and 1080 px ("Fine-tune" wrapped). **Root cause:** the nav links only hid below 860 px. **Fix:** they now hide below 1080 px, where the parts rail also folds.
- **Verify:** `node scripts/verify.mjs`: PASS. Headless Chrome at 1600 px, 900 px and 360 px mobile: ankh visible and lit, no crowding.

## 2026-10-27: ITER: Menu button reads "Change Font & Theme"

- **What:** At Joona's request the appearance button now says **Change Font & Theme**, since the menu holds both. README updated.
- **Verify:** `node scripts/verify.mjs`: PASS. Headless Chrome at 1600 px (one line) and 360 px mobile (wraps to two lines inside the pill, no overlap).

## 2026-10-27: ITER: Parts rail pinned to the window's left edge

- **What:** Joona: the boxed parts rail felt claustrophobic; it should sit on the left edge. The rail is now a 300 px sticky column flush with the window's left edge, full height, with its own divider and a faint panel tint; the reading column (740 px) is centred in the space to its right. Below 1080 px it still folds into the Contents dropdown.
- **Verify:** `node scripts/verify.mjs`: PASS. Headless Chrome at 1600 px and 1120 px: rail flush left, text centred in the remaining width.

## 2026-10-27: ITER: Layout overhaul: part cards in the contents rail, one shared grid

- **What:** Joona: lots of empty space on the left and the layout "seems off". **Root cause of the "off":** the hero sat left-aligned in a 1240 px box while the body grid (200 px rail + 700 px column) was `justify-content: center`, so the hero, rail and text started at different x positions, and the slim rail left the left side empty. **Fix:** one grid for everything (`--max` 1320 px, rail 280 px, column 740 px, left-aligned), so the hero, rail and text share one left edge. The rail now shows the four parts plus Reference as numbered cards (I–IV, §) in the display font; the card for the part you're reading opens to list its sections, with the current one highlighted (`base.js` scrollspy sets `.current`). The phone contents list shows every part's sections.
- **Verify:** `node scripts/verify.mjs`: PASS, 320 checks. Headless Chrome screenshots at 1600 px (hero; start; deep in Part 1 with its card open), 1120 px and 500 px.

## 2026-10-27: ITER: Basalt reads in Cinzel

- **What:** Joona picked Cinzel (the gold tagline face) as Basalt's font. Basalt's default **Serif** setting now sets body text in Cinzel as well as headings; Sans and OpenDyslexic still switch the body. New readers default to Serif again (not OpenDyslexic). Other themes unchanged.
- **Verify:** `node scripts/verify.mjs`: PASS, 310 checks. Headless Chrome, fresh profile: `data-theme="basalt"`, `data-font="serif"`; hero and scroll-7000 screenshots show Cinzel body and headings.

## 2026-10-27: ITER: Basalt theme, now the default (lovespark.love's look)

- **What:** Joona asked for the guide to match lovespark.love: a new **Basalt** theme (carved ash stone `#16171a`, ember gold `#d9a23b`, the site's radial stone ground, sepia mascot, stone-and-gold buttons) with the site's type: **Cinzel** headings, **OpenDyslexic** body, **DM Mono** labels. Basalt is first in the menu and is the default for new readers, who also default to OpenDyslexic. Readers with a saved theme or font keep it. Cinzel and DM Mono are inlined in `template/fonts.css` (SIL OFL 1.1) so the page stays offline and self-contained. The four other themes are unchanged.
- **Verify:** `python3 scripts/build.py` then `node scripts/verify.mjs`: PASS, 310 checks. Local Chromium with storage cleared: `data-theme="basalt"`, `data-font="dyslexic"`, Cinzel and DM Mono loaded, headings in Cinzel, labels in DM Mono. Headless Chrome (CPU and GPU) screenshots at the hero, scroll 7000 and scroll 60000 render correctly; hero also checked at narrow width.
- **Note:** the desktop app's browser pane stops painting this page (and the live guide with OpenDyslexic) once scrolled, while real Chrome paints it fine; that's a pane capture limit, not a page bug.

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
