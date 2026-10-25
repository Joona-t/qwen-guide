# Qwen, locally

A first-principles, **offline** guide to local Qwen models, in four parts: **run** one on a Mac, Linux or Windows PC; **build on it** with retrieval and memory shared between agents; **fine-tune** it with LoRA; and **understand abliteration**, what it costs and what you take on. No paid API anywhere.

Read it at **[lovespark.love/qwen-guide](https://lovespark.love/qwen-guide/)**, or download the repo and open **`index.html`** in any browser. It's one self-contained file: fonts and the mascot are inlined, it makes no network requests, and nothing leaves your machine. Prefer plain text? [`source/qwen-guide.md`](source/qwen-guide.md) has the same content.

It absorbed the earlier **Qwen Local** guide (a 16 GB Mac build companion) in October 2026: its retrieval, vector-store, shared-memory and hardening material is now Part 2, and its three calculators live on here.

## What's in it

**Part 1 — Run Qwen locally.** What a local model actually is. Whether local suits your use case, and how Qwen compares with Gemma, gpt-oss, Ministral, Granite and Llama. Which Qwen3.5 size for which job. What Hugging Face is and how to use it: model cards, file formats, the `hf` command, licenses. Installing Ollama on all three platforms, GPU requirements, first run, thinking mode, other runtimes, using the model from code, and keeping it on your machine. A decode-speed calculator.

**Part 2 — Build on it.** Embeddings and rerankers, retrieval over your own notes (RAG) and the upgrades past naive top-k, a Qdrant vector store, memory shared between agents over MCP, the security record and hardening, routing work across a local box, a GPU machine and a hosted model, and a week-by-week build order. A 16 GB budget calculator and a compounding-reliability calculator.

**Part 3 — Fine-tune.** When fine-tuning is the right tool. An interactive single-weight trainer, a LoRA parameter counter and a weight-storage calculator. What fits in 8 to 32 GB. A decide-first checklist, a dataset checklist, the Mac path (`mlx-lm`) and the NVIDIA path (Unsloth, with export to Ollama), honest evaluation, serving, and a quiz.

**Part 4 — Abliteration.** How refusal removal works, what later research found, the measured capability cost, a full responsibility-and-safety section with a decision check, the three local routes (published models, Heretic, by hand), and how to evaluate the result.

Each part ends with a **Learn more** list. A **Reference** section has troubleshooting, sources and caveats. Every command and number was checked against a primary source in **October 2026**; anything that isn't is labelled *secondary* or *untested*.

## Editing

`source/qwen-guide.md` is the source of truth. `index.html` is generated from it:

```bash
python3 scripts/build.py      # regenerate index.html
node scripts/verify.mjs       # build freshness, widget math, anchors, no network, no personal paths
```

Parts, sections and widgets are placed with HTML comments inside the Markdown (`<!-- part ... -->`, `<!-- section ... -->`, `<!-- widget:name -->`), so the file still reads cleanly on GitHub. Widgets, styles (`template/style.css`), the bundled font and the page script are in `template/`.

## Design and accessibility

The page uses an editorial layout modelled on the Raw Thought reader: flat paper, hairline rules, a serif reading column and a slim contents rail. The **Change Theme** menu has four themes (Bubblegum, Retro Pink, Dark Noir, Beige Paper) and three reading fonts. **Serif** and **Sans** use system fonts. **OpenDyslexic** is bundled in the file. Every theme's text colours pass WCAG AA, at 4.8:1 or better. The page supports full keyboard navigation, shows visible focus rings, honours `prefers-reduced-motion`, and uses ARIA live regions on the calculators. It works down to 320 px.

## License

MIT. Qwen3.5 model weights are Apache-2.0 and are not included here.

100% local · no paid API · made for neurodivergent builders 🐺
