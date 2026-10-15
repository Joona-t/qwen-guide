# Bugs & Iterations — Qwen, locally

Newest first.

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
