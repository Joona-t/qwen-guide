# Qwen fine-tuning guide

A first-principles, **offline** guide to fine-tuning Qwen3.5 locally with LoRA, on an **Apple Silicon Mac** (`mlx-lm`) or an **NVIDIA PC** (Unsloth). No paid API anywhere.

It's the companion to [Qwen Local](https://github.com/Joona-t/qwen-local-guide), which covers running Qwen, RAG and shared agent memory on a 16 GB Mac.

## Open it

Download the repo and open **`index.html`** in any browser. It's one self-contained file: fonts and the mascot are inlined, it makes no network requests, and nothing leaves your machine. Prefer plain text? Read [`source/qwen-fine-tuning-guide.md`](source/qwen-fine-tuning-guide.md), which has the same content.

## What's in it

- **Foundations.** What fine-tuning changes, and how it compares with prompting, retrieval and better tools.
- **One learning step, made visible.** An interactive single-weight trainer that shows why lower loss isn't the same as "better."
- **LoRA and QLoRA.** A live parameter counter, and why QLoRA isn't recommended for Qwen3.5.
- **Will it fit?** A weight-storage calculator, how to read your real memory budget, and a fits table for Macs (16/24/32 GB+) and NVIDIA cards (8/12/16/24 GB).
- **Decide first, then build the dataset.** Baseline, held-out test, pass rule, stop rule, data checklist, thinking-mode handling.
- **Path A, Mac:** `mlx-lm` install → train → test → serve, plus the Qwen3.5 hybrid-layer gotcha.
- **Path B, NVIDIA:** Unsloth on Linux, WSL or Windows → train → GGUF export → Ollama.
- **Evaluate, ship, go further.** Honest evaluation, serving, DPO/GRPO, vision, and a pointer to abliteration.
- **Quiz, troubleshooting table, sources, caveats.**

Every command and number was checked against a primary source in **October 2026**. Anything that isn't is labelled *secondary* or *untested*.

## Editing

`source/qwen-fine-tuning-guide.md` is the source of truth. `index.html` is generated from it:

```bash
python3 scripts/build.py      # regenerate index.html
node scripts/verify.mjs       # build freshness, widget math, anchors, no network, no personal paths
```

Section and widget placement live in HTML comments inside the Markdown (`<!-- section ... -->`, `<!-- widget:name -->`), so the file still reads cleanly on GitHub. Widgets, styles and the page shell are in `template/`.

## Accessibility

OpenDyslexic body font with a system fallback, four themes (Candy, Retro Pink, Dark Noir, Beige Paper), full keyboard navigation, visible focus rings, `prefers-reduced-motion` support and ARIA live regions on the calculators. Responsive down to 320 px.

## License

MIT. Qwen3.5 model weights are Apache-2.0 and are not included here.

100% local · no paid API · made for neurodivergent builders 🐺
