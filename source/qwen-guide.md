# Qwen, locally: run it, fine-tune it, understand abliteration

> Written **October 2026** for Qwen3.5 (0.8B / 2B / 4B / 9B), on an Apple Silicon Mac or a PC with an NVIDIA GPU. Every command, version and number was checked against a primary source that month: Ollama's and Hugging Face's documentation, PyPI, the `mlx-lm` and Unsloth docs, model cards, arXiv. Anything else is labelled *secondary* (a blog or forum) or *untested*. Fast-moving facts go stale, so re-check versions before you rely on them.

**Who this is for.** Anyone who wants a capable language model on their own machine and wants to understand what they're running. No paid API is involved anywhere, and nothing you do here leaves your computer.

**How it's organised.** Three parts, in the order you'd do them:
- **Part 1 — Run Qwen locally.** What a local model is, whether local suits your use case, how Qwen compares with other families, what Hugging Face is, and how to install and run a model on a Mac, Linux or Windows.
- **Part 2 — Fine-tune.** When fine-tuning is the right tool, how LoRA works, what fits in your memory, how to build a dataset, train on a Mac or an NVIDIA PC, and evaluate honestly.
- **Part 3 — Abliteration.** What refusal removal is, what the research found, what it costs, the responsibility you take on, and how people do it locally.

Each part ends with a **Learn more** list of the resources it was built from. The **Reference** section at the end has troubleshooting, sources and caveats.

**Three rules this guide keeps coming back to:**
1. **Run it before you change it.** Most "the model can't do X" problems are solved with a better prompt, retrieval or a different model, not with training or surgery.
2. **Test on your own task.** A small set of real inputs with the answers you'd accept, written before you start, beats every leaderboard and every training-loss curve.
3. **Own what you ship.** A local model is yours: its privacy, its mistakes, and anything you removed from it.

<!-- part title="Part 1 · Run Qwen locally" -->
<!-- section id="run-basics" toc="What a local model is" kicker="Part 1 · Run Qwen locally" -->
## What you're actually running

A language model is a file of numbers, called weights, plus a small program that uses them to predict the next token of text, one token at a time. Running it locally means that file sits on your disk and that program runs on your own CPU or GPU. Nothing leaves the machine unless you send it somewhere.

Three facts decide almost everything about running one well:

**Memory sets the ceiling.** The whole model has to be in memory while it runs. The size on disk is roughly `parameters × bits ÷ 8`: a 4B model at 16-bit is about 9 GB; at 4-bit, about 2.5 GB. Add a gigabyte or two for the working memory that grows with the length of the conversation (the KV cache).

**Memory bandwidth sets the speed.** For every generated token, the program reads the entire model from memory. So tokens per second is roughly memory bandwidth divided by model size. A Mac mini with the M4 chip (2024) reads about 120 GB/s, so a 6.6 GB model tops out near 18 tokens per second there; the current base Mac mini (M6) reads 153 GB/s, which lifts that ceiling to about 23. Real runs land a bit below either figure. A discrete GPU is several times faster because its memory is.

**Quantization is what makes it fit.** Models are trained in 16-bit. Storing the weights in 4-bit cuts the file to about a quarter with a small quality loss. `Q4_K_M` is the common default. Sizes like `Q8_0` keep more quality at twice the size. Below 4-bit, quality drops quickly. A smaller model at 4-bit is almost always better than a bigger model crushed to 2-bit.

> 📝 **Rule of thumb.** Take the model's parameter count in billions and multiply by 0.6. That's roughly the gigabytes a 4-bit copy needs, before the working memory. Then leave a quarter of your RAM free for the operating system.

<!-- section id="use-case" toc="Local models and your use case" kicker="Part 1 · Run Qwen locally" -->
## Local models as a whole: is local right for your use case?

This guide is about Qwen, but the first decision is bigger than any one model family. Running a model locally is a trade. You give up the strongest models and the convenience of a hosted API. You get privacy, no per-token cost, no rate limits, offline use, and a system you fully control. Whether that trade is worth it depends on what you're doing.

**Local is a clear win when:**
- the data is private: health notes, client files, a codebase you can't upload, your own journal;
- the work is high-volume and repetitive: classifying thousands of emails, tagging a photo library, summarising transcripts, where API costs add up;
- you need it offline, or on a schedule, or inside a tool that must keep working when a vendor changes its terms;
- latency matters more than depth: autocomplete, quick rewrites, extraction into JSON;
- you're learning: nothing teaches how these systems work like running, breaking and tuning one yourself.

**A hosted frontier model is still the right call when:**
- the task needs the best reasoning available: hard maths, long multi-step agent work, subtle judgement calls where being wrong costs real money;
- you need very long documents handled reliably in one pass;
- you'll use it rarely. The setup cost of local isn't worth it for ten prompts a month.

Many good setups use both: a local model for the private, repetitive and fast work, and a hosted model for the hard calls. The companion [Qwen Local guide](https://github.com/Joona-t/qwen-local-guide) describes a three-tier version of this.

**Write your use case down in one sentence** before choosing anything. "Summarise my meeting transcripts into action items, offline, on my laptop" points at a different model and runtime than "help me write Rust in my editor" or "answer questions over 2,000 PDFs."

### The local landscape, October 2026

Qwen is one family among many, and "open weights" no longer means "runs on your laptop." The smallest current GLM-5.x is 320B parameters; Kimi K3 is 2.8 trillion. So this section is two tables: what fits a laptop, and the open-weight frontier you reach through an API. Every row was checked against the publisher's model card or Ollama's library page on **2026-10-15**. The sizes are 4-bit downloads as Ollama or the official GGUF repo lists them.

**Fits a laptop** (16 GB with headroom, or 24 GB):

| Family | Publisher | Laptop sizes (4-bit) | License | Released | Notes |
|---|---|---|---|---|---|
| **Qwen3.5** 0.8B / 2B / 4B / 9B | Alibaba | 1.2 / 2.7 / 3.3 / 6.6 GB; 27B at 17 GB needs 24 GB | Apache-2.0 | Feb–Mar 2026 | 200+ languages, text + image input, 256K context, best small tool-caller. This guide's pick for 16 GB. |
| **Qwen3.8-27B** | Alibaba | 18 GB, 24 GB machines only | Apache-2.0 | Aug 2026 | The newest Qwen that fits anything portable: dense, text + image, 256K. Supersedes Qwen3.5-27B and Qwen3.6-27B (17 GB) on a 24 GB machine. |
| **Gemma 4** E2B / E4B / 12B / 26B-A4B / 31B | Google | 4.3–4.6 / 6.1–6.6 / 7.2–8.0 GB; 26B-A4B and 31B at 16–20 GB need 24 GB | Apache-2.0 | Mar–Jun 2026 | E2B, E4B and the 12B take text, image *and audio*; 26B-A4B and 31B are text + image. 128K context on the small ones, 256K above. Quantization-aware checkpoints hold quality at 4-bit. |
| **gpt-oss-20b** | OpenAI | 14 GB; tight on 16 GB, fine on 24 GB | Apache-2.0 | Aug 2025 | 21B total, 3.6B active, adjustable reasoning effort, text only, 128K. Still OpenAI's only laptop-size open model; no successor as of Oct 2026. |
| **Ministral 3** 3B / 8B / 14B | Mistral | 3.0 / 6.0 / 9.1 GB | Apache-2.0 | Dec 2025 | Base, Instruct and Reasoning at each size, 256K, text + image, function calling. Mistral's 2026 open releases (Small 4 at 119B, Medium 3.5 at 128B) are too large for a laptop, so this is still its laptop line. |
| **Muse Glimmer 30B** | Meta | 18 GB, 24 GB machines only | Apache-2.0 | Aug 2026 | Meta's current open model, distilled from its closed Muse Spark; ~29.6B dense, text + image, 131K. Meta has shipped no Llama since Llama 4 (Apr 2025), whose smallest member is 67 GB at 4-bit. |
| **Granite 4.2** 3B / 8B / 30B | IBM | 2.2 / 5.3 GB; 30B at 17–18 GB needs 24 GB | Apache-2.0 | Aug 2026 | Dense reasoning family with thinking and non-thinking modes, 128K native, tool calling. Built for retrieval and tool use; small and well-behaved. |
| **LFM2.5** 8B-A1B / 24B-A2B | Liquid AI | 5.2 GB; 24B-A2B at 15 GB needs 24 GB | LFM Open License 1.0 (custom) | May–Oct 2026 | Sparse mixture-of-experts built for speed on CPUs and laptops (1.5B and 2.45B active), 128K, text only. The 24B-A2B shipped 2026-10-12, the newest laptop-class model in this table. |
| **Nemotron 3 Nano 4B** | NVIDIA | 2.8 GB | NVIDIA Nemotron Open Model License | Mar 2026 | 256K context, tool calling. The 30B-A3B sibling is 24 GB at 4-bit and needs a 32 GB machine. Licenses differ across the Nemotron line, so read each card. |
| **Phi-4** mini 3.8B / 14B / reasoning-vision 15B | Microsoft | 2.5 / 9.1 GB; vision model ~10 GB + projector | MIT | Dec 2024 – Mar 2026 | The reasoning-vision 15B (Mar 2026, 16K context) is the newest Phi; there is no Phi-5 as of Oct 2026. |
| **GLM-4.7-Flash** 30B-A3B | Z.ai (Zhipu) | 19 GB, 24 GB machines only | MIT | Jan 2026 | The only GLM with a local Ollama tag; 198K context, text, tools and thinking. Everything in the GLM-5.x line is far too large (see below). |
| **DeepSeek-R1-0528-Qwen3-8B** | DeepSeek | 5.2 GB | MIT | May 2025 | The only DeepSeek-branded model that fits a laptop: R1 reasoning distilled onto Qwen3-8B, text only, 128K. Every DeepSeek V4-generation model is 284B or larger. |
| **Qwen2.5-Coder** 7B / 14B, **Qwen3-Coder-30B-A3B** | Alibaba | 4.7 / 9.0 GB; 30B-A3B at 19 GB needs 24 GB | Apache-2.0 | Nov 2024 / Jul 2025 | Both document fill-in-the-middle, which newer agentic coders (Qwen3.8-27B, Granite 4.2, Devstral Small 2) don't, so they remain the editor-autocomplete picks. |
| **SmolLM3-3B** | Hugging Face | 1.9 GB (GGUF from ggml-org; not in Ollama's library) | Apache-2.0 | Jul 2025 | Fully open training recipe, 128K, strong tool calling for its size. A reference model rather than a daily driver. |

**The open-weight frontier** (you can download these, but you run them through an API or a datacenter):

| Model | Publisher | Size | License | Released | Why a laptop user should know it |
|---|---|---|---|---|---|
| **Kimi K3** | Moonshot AI | 2.8T total, 104B active, 896 experts | Kimi K3 License (custom, MIT-derived) | Jul 2026 | The largest open-weight model that exists: 1M context, text + image. About 1.5 TB at 4-bit. Shows what "frontier" means now. |
| **Qwen3.8-2.4T-A95B** | Alibaba | 2.4T total, 95B active | Qwen3.8-Max License (custom) | Aug 2026 | The open checkpoint of hosted Qwen3.8-Max, text only, thinking always on. Same generation as the laptop-size Qwen3.8-27B. |
| **DeepSeek-V4.1-Flash** | DeepSeek | 552B backbone, 8B/16B active | MIT | Sep 2026 | DeepSeek's current release: text + image, 1M context. The model its API now serves for everything. |
| **GLM-5.3** | Z.ai (Zhipu) | 753B (about 40B active, per the GitHub README) | GLM-5.3 License (custom) | Aug 2026 | Z.ai's flagship coding and agent model, text only, 1M context. The first GLM to leave the MIT license. |
| **GLM-5.3-Flash** | Z.ai (Zhipu) | 320B total, 18B active | MIT | Aug 2026 | Natively multimodal. "Flash" no longer means laptop-class: about 200 GB at 4-bit. |
| **Mistral Large 4** | Mistral | 1T total, 52B active | not stated | Oct 2026 | Announced 2026-10-06 with weights promised "by the end of the month." Not open as of 2026-10-15. Mistral Large 3 (675B, Apache-2.0, Dec 2025) is the current download. |
| **MiniMax-M3** | MiniMax | ~428B total, ~23B active | MiniMax Community License (custom; restrictions on commercial use) | Jun 2026 | Natively multimodal, 1M context. Its license is stricter than the earlier M2 line's. |
| **gpt-oss-120b** | OpenAI | 117B total, 5.1B active | Apache-2.0 | Aug 2025 | 65 GB at 4-bit: a 64 GB-plus Mac or an 80 GB GPU. The big sibling of the laptop-size 20B. |
| **Llama 4** Scout / Maverick | Meta | 109B / 400B total, 17B active | Llama 4 Community License | Apr 2025 | Still the newest Llama. Scout is 67 GB at 4-bit. Useful mainly as the anchor for Llama-format tooling. |
| **Nemotron 3 Ultra** | NVIDIA | 550B total, 55B active | OpenMDW 1.1 | Jun 2026 | NVIDIA's frontier open model with open data and recipes. Its license differs from the Nano line. |
| **Mistral Small 4** | Mistral | 119B total, 6.5B active | Apache-2.0 | Mar 2026 | Despite "Small," about 58–74 GB at 4-bit. Merges Mistral's reasoning, vision and coding lines. |
| **Qwen3-Coder-Next** | Alibaba | 80B total, 3B active | Apache-2.0 | Feb 2026 | Fill-in-the-middle agentic coder at 52 GB: runs on a 64 GB desktop or Mac Studio, not a laptop. |

**Why the frontier matters even if you can't run it.** Three reasons. The frontier models are the API fallback for the hard calls a local model can't make. Their labs often ship a small sibling trained the same way (Qwen3.8-27B next to the 2.4T, Gemma 4 E2B next to the 31B). And their licenses are drifting away from MIT and Apache toward custom terms with revenue clauses, which matters the moment you build something on one.

> 📝 **Watch list, unverified on 2026-10-15.** Qwen4 is described as "in training" in secondary reports, with no date. NVIDIA announced a Nemotron 3.5 Bolt 74B-A9B for mid-October. JetBrains published Mellum2.1, a 12B-A2.5B coding model, on 2026-10-07 with an official GGUF at 7–8 GB; its benchmark numbers are self-reported. Treat all three as leads until their model cards exist and you've read them.

There are more families than this, and new releases arrive weekly. Treat any table like this one, including this one, as a dated snapshot, and read the model card before you commit to a family.

### How to choose

1. **Memory first.** List what you have: RAM on a Mac, GPU memory on a PC. Subtract a quarter for the system. That number eliminates most of the table before you think about quality.
2. **Task type.** Tool calling and structured output favour Qwen3.5-9B. Reasoning-heavy chat favours gpt-oss-20b if it fits. Editor autocomplete needs fill-in-the-middle (Qwen2.5-Coder). Images need a vision-capable model (Qwen3.5, Gemma 4, Ministral 3).
3. **Language.** If you work in a language other than English, Qwen's multilingual training is a real advantage. Check the model card for the languages it lists.
4. **License.** Apache-2.0 means no surprises. Other licenses may restrict commercial use, user counts or redistribution.
5. **Ecosystem.** Does your runtime support it today? Does the fine-tuning tool? Do community quantizations exist? A slightly weaker model with full support beats a stronger one you can't run.
6. **Then test it on your task.** Write ten to thirty real inputs with the answers you'd accept. Run each candidate with the same prompt. Score them. This takes an hour and it's the only step that measures what you care about. Part 2 reuses exactly this test set if you go on to fine-tune.

> 📝 **Benchmarks are a starting point, not a verdict.** Public leaderboards measure general tasks on public data, and some have known defects. A model that scores five points higher on a benchmark can be worse on your inputs. Your own small test set outranks every leaderboard for your use case.

<!-- section id="models" toc="Which Qwen" kicker="Part 1 · Run Qwen locally" -->
## Which Qwen?

Qwen is Alibaba's family of open-weight models. The useful line for a personal machine in October 2026 is **Qwen3.5**, released in February and March 2026. Its small models are **0.8B, 2B, 4B and 9B**. All are Apache-2.0 licensed, take text and images as input, and have a 256K-token context window. Newer lines (Qwen3.6, Qwen3.8) only ship at 27B and larger, so Qwen3.5 is still the right pick for a 16 GB machine. On a 24 GB machine, Qwen3.8-27B (18 GB at 4-bit) is the newer choice.

| Model | 4-bit size on Ollama | Fits comfortably in | Good for |
|---|---|---|---|
| Qwen3.5-0.8B | 1.2–1.3 GB | anything, even a phone-class device | classification, quick drafts, the draft model for speculative decoding |
| Qwen3.5-2B | 2.7–3.1 GB | 8 GB | fast extraction and summaries; first fine-tuning experiments |
| Qwen3.5-4B | 3.3–4.0 GB | 8 GB, with headroom at 16 GB | the everyday local assistant; strong on structured output |
| Qwen3.5-9B | 6.6–7.6 GB | 16 GB | the quality pick; best local tool-calling at this size |
| Qwen3.5-27B | 17–20 GB | 24 GB | dense, not mixture-of-experts; on a 24 GB machine prefer Qwen3.8-27B (18 GB, Aug 2026) |

The sizes are what Ollama downloads for its default tags, which include the vision encoder. Larger Qwen3.5 models (35B-A3B, 122B-A10B, 397B-A17B) are mixture-of-experts and need workstation memory.

**Pick by task, not by ego.** Run your real prompts through the 4B first. If it does the job, you've saved memory for context and other programs. Move to the 9B when answers need more reasoning or when tool-calling has to be reliable. Tool-calling reliability drops sharply below 9B: on the BFCL benchmark the 9B scores about 66, the 4B about 50, the 2B about 44.

> ⚠ **The 4B and 9B think by default.** They write a `<think>` block before answering. That's good for hard questions and slow for everything else. Only the 0.8B and 2B default to non-thinking. That's what the models' own chat templates do; Unsloth's Qwen3.5 page describes all four as reasoning-off by default, so check the `think` setting on your runtime rather than assuming. Section "Thinking mode" below shows how to switch it.

**Other Qwen models you'll see.** `Qwen2.5-Coder` (7B, 14B, 32B) is older but still the best Qwen for editor autocomplete, since it supports fill-in-the-middle. `Qwen3-VL` is a vision line. `Qwen3-Embedding-0.6B` and `Qwen3-Reranker` are for search and retrieval. `Qwen3Guard` is a safety classifier. None of these are needed to get started.

<!-- section id="huggingface" toc="Hugging Face" kicker="Part 1 · Run Qwen locally" -->
## Hugging Face: where the models live

**What it is.** [Hugging Face](https://huggingface.co/) is a company and a website, usually called "the Hub." It hosts open model weights, datasets and demos the way GitHub hosts code. It also maintains the Python libraries most of the ecosystem is built on: `transformers` for running models, `datasets` for data, `peft` for LoRA adapters and `trl` for fine-tuning. When someone says "the model is on Hugging Face," they mean a Git-like repository on the Hub containing the weight files.

**Reading a model page.** A model repo such as [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) has three parts worth reading:
- **The model card** is the README. It states the license, the architecture, the context length, benchmark numbers and how to run it. Read the license first.
- **Files and versions** lists the actual weights. The sidebar's parameter count is sometimes wrong for packed formats, so trust the card, not the sidebar.
- **The organisation** tells you who published it. `Qwen` is the official account. `unsloth`, `mlx-community`, `bartowski` and `mradermacher` are well-known community accounts that re-package official weights into other formats. Anyone can upload anything, so treat an unfamiliar account the way you'd treat an unfamiliar download site.

**The three file formats you'll meet.**
- **safetensors** is the standard format for full-precision weights. It's what Hugging Face libraries, Unsloth and Heretic load, and it can't execute code when opened. Prefer it over the older `.bin` or `.pt` pickle files, which can.
- **GGUF** is the format for llama.cpp, Ollama and LM Studio. One file per quantization level: `Q4_K_M`, `Q8_0`, `F16`. This is what you run on a laptop.
- **MLX** repos (usually under `mlx-community`) are weights converted for Apple's MLX framework, in 4-bit, 8-bit or bf16. This is what `mlx-lm` loads on a Mac.

The official `Qwen` account publishes safetensors. The GGUF and MLX versions come from the community accounts above, and the Qwen docs point to them.

**The `hf` command.** Hugging Face's command-line tool is called `hf`. Install it with one of:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash      # macOS and Linux
brew install hf                                      # macOS with Homebrew
pip install -U huggingface_hub                       # any Python environment
```
On Windows, use PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"`. Then:
```bash
hf download Qwen/Qwen3.5-4B                                   # whole repo into the cache
hf download unsloth/Qwen3.5-4B-GGUF --include "*Q4_K_M*" --local-dir ./models   # one quant, into a folder
hf download Qwen/Qwen3.5-4B config.json                       # a single file
```
Downloads go to a cache under `~/.cache/huggingface/hub` unless you pass `--local-dir`. Big models fill disks quickly, so know where that folder is.

**Accounts and tokens.** You don't need an account to download public models like Qwen. Some models are "gated," which means the publisher asks you to accept terms on the model page first. Those, and private repos, need `hf auth login` with a token from your account settings. Never paste a token into a shared notebook or commit it to a repo.

**Running a Hub GGUF straight from Ollama.** You don't have to download by hand. Ollama can pull a GGUF repo directly:
```bash
ollama run hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```
The part after the colon picks the quantization file from the repo; without it Ollama takes `Q4_K_M` when the repo has one. Each GGUF model page on the Hub also offers an "Use this model → Ollama" menu that writes the command for you. This is handy for community builds that aren't in Ollama's own library, including fine-tuned and abliterated variants, and it's exactly where the trust question above matters most.

**Licenses.** Qwen3.5's small models are Apache-2.0: you can use, modify and redistribute them, including commercially, if you keep the license notice. Other model families differ. Llama has its own community license. Gemma 4 and Meta's Muse Glimmer are Apache-2.0. Licenses diverge inside the Qwen3.8 line: the 27B is Apache-2.0, the 2.4T-A95B uses the Qwen3.8-Max License, and Flash-Next uses the Qwen Community License 1.0. GLM-5.3, Kimi K3 and MiniMax-M3 ship under custom licenses rather than MIT. The model card tells you. Derivatives you publish inherit the base model's terms, so name the base model in your own card.

<!-- section id="install" toc="Install Ollama" kicker="Part 1 · Run Qwen locally" -->
## Install Ollama and run your first Qwen

[Ollama](https://ollama.com/) is the easiest runtime on all three platforms. It wraps llama.cpp (and, on Apple Silicon, Apple's MLX: a preview since v0.19 in March 2026, the default for supported models since v0.40.0 in September), serves a local HTTP API on port 11434, and manages model files for you.

**Install:**
```bash
brew install ollama                                  # macOS (or download the app from ollama.com)
curl -fsSL https://ollama.com/install.sh | sh        # Linux
```
On Windows, download the installer from [ollama.com/download/windows](https://ollama.com/download/windows). The macOS and Windows apps run the server in the background. On Linux, the installer sets up a service; if it isn't running, start it with `ollama serve`.

**GPU requirements.** NVIDIA GPUs need compute capability 5.0 or newer and driver 550 or newer (570 or newer for compute capability 5.0 to 6.2). AMD GPUs need the ROCm v7 driver on Linux, or the ROCm v7 / HIP7 stack on Windows for RX 7000-series and PRO W7500–W7900 cards. Other GPUs on Windows can use Vulkan. Apple Silicon uses Metal with no setup. If Ollama can't see a GPU it falls back to the CPU, which works but is slow.

**First run:**
```bash
ollama pull qwen3.5:4b
ollama run qwen3.5:4b "Explain unified memory in one paragraph."
```
Always write the tag. A bare `ollama run qwen3.5` pulls the **9B**, about 7 GB, because `latest` points at it.

**Check that it works:**
```bash
ollama --version                              # expect the latest release (v0.40.x in Oct 2026)
ollama list                                   # models on disk
ollama ps                                     # what's loaded now, and whether it's on the GPU
curl -s http://127.0.0.1:11434/api/tags       # the server answers on loopback
```
Watch your system's memory monitor during the first run. If the machine starts swapping, the model is too big for comfort. Drop to a smaller one.

**Housekeeping.** Models live under `~/.ollama` (Linux and macOS) or in your user profile on Windows. `ollama rm qwen3.5:9b` deletes one. Budget 10–12 GB of disk for a 4B, a 9B and an embedding model.

**Settings for a 16 GB Mac** (set with `launchctl setenv`, then quit and reopen Ollama from the menu bar):
```bash
launchctl setenv OLLAMA_HOST 127.0.0.1:11434   # loopback only; never 0.0.0.0
launchctl setenv OLLAMA_FLASH_ATTENTION 1      # smaller KV cache; needed for the next line
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0     # halves KV-cache memory, tiny quality cost
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1    # one model resident at a time
launchctl setenv OLLAMA_NUM_PARALLEL 1         # don't multiply the KV cache
launchctl setenv OLLAMA_KEEP_ALIVE 5m          # unload after five idle minutes
```
On Linux, put the same variables in the service's environment; on Windows, set them as user environment variables and restart Ollama. On a box with more memory, the first line is the only one that matters.

> ◆ **Ollama's MLX runner on a 16 GB Mac is undocumented.** Since v0.40.0, Ollama runs supported models (Qwen3.5, Qwen3.6, Qwen3.8 and Gemma 4 included) on Apple's MLX by default. The March 2026 preview post said MLX wanted more than 32 GB of unified memory; the v0.40.0 release notes state no memory requirement. What a 16 GB Mac does is not documented. Check `ollama ps` and the server log to see which runner loaded, and compare speed against a `-q4_K_M` tag if in doubt.

<!-- section id="thinking" toc="Thinking mode" kicker="Part 1 · Run Qwen locally" -->
## Thinking mode

Qwen3.5 4B and 9B reason out loud in a `<think>` block before answering. It helps with maths, planning and tricky instructions. It costs time, and quantized models in thinking mode truncate answers more often because they think for longer. For extraction, classification, chat and tool calls, turn it off.

- **In the terminal:** `ollama run qwen3.5:4b --think=false`, or type `/set nothink` inside a session. `--hidethinking` keeps the reasoning but hides it.
- **Through the API:** send a top-level `"think": false` field on `/api/chat` or `/api/generate`. Inside `options` it is silently ignored.
- **In a Modelfile:** `PARAMETER think false` is **not supported** (Ollama issue #10961). Set it per request.
- **In Python, MLX or llama.cpp:** pass `enable_thinking=False` to the chat template. For llama.cpp's server: `--chat-template-kwargs '{"enable_thinking":false}'`.
- **Sampling:** Unsloth's Qwen3.5 page suggests temperature 1.0, top_p 0.95, top_k 20 with thinking on, and 0.7 / 0.8 / 20 with it off.

Decide which mode you'll use for a task and stay consistent, including when you evaluate (Part 2) or compare models (Part 3).

<!-- section id="runtimes" toc="Other runtimes" kicker="Part 1 · Run Qwen locally" -->
## Other runtimes, and when to pick them

- **[LM Studio](https://lmstudio.ai/)** is a desktop app with a model browser, a chat window and an OpenAI-compatible server. On a Mac it can run both llama.cpp and MLX engines and supports speculative decoding. The best choice if you'd rather click than type.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** is the C++ engine underneath Ollama and LM Studio. Use it directly when you need every option: exact GPU layer offload when a model barely fits, a reranking endpoint, or the latest model support before the apps catch up. `llama-server` gives you an OpenAI-compatible API.
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** is Apple's Python package for MLX. It's the fastest way to run small models on a Mac for short prompts, and the only Mac path for LoRA training (Part 2). It loads the whole model into GPU memory or nothing, and slows down at very long contexts.
- **vLLM, SGLang** and similar servers are for GPU boxes serving many users at once. Not for a laptop.

For one person on one machine, start with Ollama. Add the others when a specific need appears.

<!-- section id="api" toc="Use it from code" kicker="Part 1 · Run Qwen locally" -->
## Use it from your own code

Every runtime above speaks the OpenAI chat-completions format, so any client library works by changing the base URL. With Ollama running:
```bash
curl -s http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:4b","messages":[{"role":"user","content":"Say hi"}]}'
```
In Python, with the `openai` package: create the client with `base_url="http://127.0.0.1:11434/v1"` and any non-empty `api_key`, then call `chat.completions.create` as usual. Ollama's own `/api/chat` endpoint adds the `think` switch and structured-output support (`format` with a JSON schema), which are worth using for extraction tasks.

Context length is set per model, not per request. In Ollama, create a Modelfile with `FROM qwen3.5:4b` and `PARAMETER num_ctx 8192`, then `ollama create my-qwen -f Modelfile`. Keep it modest on a 16 GB machine; the KV cache grows with it.

<!-- section id="safe-run" toc="Keep it safe" kicker="Part 1 · Run Qwen locally" -->
## Keep it on your machine

Local inference is private by default. Keep it that way:

- **Loopback only.** Ollama, LM Studio and llama.cpp have **no authentication**. Bound to `127.0.0.1` they're reachable only from your own computer. Bound to `0.0.0.0` they're open to anyone on the network, and if a router forwards the port, to the internet. Ollama has had real remote-code-execution and memory-leak vulnerabilities, and scans regularly find hundreds of thousands of exposed servers.
- **Run the latest version.** Model-file parsers are a recurring source of memory-safety bugs. Updates fix them.
- **Pull from accounts you trust.** A model file is input to a parser you run with your own permissions. Prefer the official library and known publishers; prefer safetensors and GGUF over pickle formats.
- **Treat model output as untrusted text.** If you wire a model to tools that can run commands, read files or send messages, put the approval step in your code, not in the prompt. A model can be talked into anything; your validator can't.
- **Reaching it from another machine** is fine over a VPN or mesh such as Tailscale, bound to that interface, with a firewall rule. Never expose port 11434 directly.

<!-- section id="learn-run" toc="Learn more" kicker="Part 1 · Run Qwen locally" -->
## Learn more: running models

- [Qwen documentation](https://qwen.readthedocs.io/en/latest/): the official guide, with pages for Ollama, LM Studio, llama.cpp and MLX, quantization and deployment. It still centres on the Qwen3-2507 line, but the runtime pages apply to 3.5.
- [Qwen3.5 model cards](https://huggingface.co/Qwen/Qwen3.5-4B) on Hugging Face: architecture, benchmarks, recommended sampling, the thinking switch.
- [Ollama docs](https://docs.ollama.com/): quickstart, GPU requirements, importing models, the API.
- [Hugging Face Hub docs](https://huggingface.co/docs/hub/index) and the [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the engine, with its own quantization and server documentation.
- [mlx-lm](https://github.com/ml-explore/mlx-lm): Apple's package for running and training on MLX.
- [The Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1): free, from tokenizers to fine-tuning, if you want the foundations.
- [Andrej Karpathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html): build a small language model from scratch in Python. The best way to understand what the weights are.
- [Qwen Local](https://github.com/Joona-t/qwen-local-guide), this guide's companion, goes further on a 16 GB Mac: embeddings, retrieval (RAG), a vector database and a shared memory layer for several agents.

<!-- part title="Part 2 · Fine-tune" -->
<!-- section id="basics" toc="What fine-tuning changes" kicker="Part 2 · Foundations" -->
## What fine-tuning actually changes

**Tokens are pieces.** A tokenizer turns text into numbered pieces: sometimes a word, sometimes part of one, punctuation or whitespace. The exact split depends on the model.

**Weights are numbers.** A model is billions of learned numbers that turn input tokens into a prediction of the next token. They are not a database of facts you can edit, and a chat does not change them. A conversation only changes the *context*.

**Training changes the numbers.** Inference uses the weights as they are. Training shows the model examples, scores its predictions against them, computes how each weight affected the error, and nudges the weights to reduce it. Fine-tuning is just more training, on your examples, starting from a model that already works.

There are four ways to change how a local model behaves. Only one of them touches the weights:

| Method | What changes | Reach for it when |
|---|---|---|
| Prompt / few-shot examples | Instructions read at inference time | Always first. It's free and instant to undo. |
| Retrieval (RAG) | Relevant documents placed in context | The model lacks *facts*: your notes, docs, codebase. |
| Better tools / validators | Code that runs and checks the output | The output must be *correct*, not just likely. |
| Fine-tuning | The model's weights (or an adapter) | The model has the facts but keeps answering in the wrong *shape*. |

> ⚠ **Fine-tuning doesn't add reliable facts.** It makes patterns more likely. A model trained on your docs will sound like your docs and still invent details. For knowledge, use retrieval. For correctness, use a validator. Fine-tune for behaviour.

Why can a pile of numbers hold behaviour at all? Because a program built from comparisons and arithmetic can be rewritten as multiplies, adds and ReLU, the same operations a network layer performs. Cheng Lou's [handwired](https://github.com/chenglou/handwired) (MIT) does this by hand for ordinary JavaScript; its [live demo](https://chenglou.me/handwired/) goes much further. The toy below takes its first example: nothing here is Qwen, but the point carries over.

<!-- widget:handwired -->

<!-- section id="step" toc="One learning step" kicker="Part 2 · Follow the numbers" -->
## One learning step, made visible

Supervised fine-tuning (SFT) shows the model examples of the answers you want. A **forward pass** predicts. A **loss** measures how wrong the prediction was. **Backpropagation** computes how each weight affected that loss. An **optimizer** moves the weights a little in the direction that lowers it. Repeat a few thousand times.

The toy below is one weight, one example and one step at a time. It isn't Qwen, but the arithmetic is the same arithmetic.

<!-- widget:gradient -->

**Loss is feedback, not truth.** Cross-entropy loss punishes the model for giving low probability to the *target*. If the target is wrong, a lower loss means the model learned the wrong thing better. Flip the toy's label to "wrong" and watch the loss fall anyway. That's why the quality of your examples matters more than their quantity.

**The learning rate is the step size.** Too small and nothing changes. Too large and the model overshoots, then forgets what it knew. One *secondary* report of a Qwen3.5 LoRA on a Mac saw a learning rate of 1e-4 *triple* the error count against the untouched base, while 1e-5 cut it by two thirds. Start small.

<!-- section id="lora" toc="LoRA and QLoRA" kicker="Part 2 · Change fewer numbers" -->
## Full fine-tuning, LoRA and QLoRA

**Full fine-tuning** updates every weight. It's flexible, but you need memory for the weights, their gradients and the optimizer state for *all* of them. Unsloth puts it at roughly 4× the memory of LoRA.

**LoRA** (Hu et al., 2021, arXiv 2106.09685) freezes the original weights `W` and trains two small matrices per targeted layer: `A` (r × d_in) and `B` (d_out × r). Their product `B·A` has the same shape as `W` and gets added to it, scaled. The rank `r` is the bottleneck: at `r = 16` on a 4096 × 4096 layer you train 131,072 numbers instead of 16.7 million. The adapter is a separate small file, so the base model stays untouched and you can switch adapters on and off.

**QLoRA** (Dettmers et al., 2023, arXiv 2305.14314) is LoRA on top of a base model stored in 4-bit. The base is smaller in memory, and the adapters still train in higher precision.

<!-- widget:lora -->

> ◆ **Scale conventions differ between tools.** The standard formula scales the update by `alpha / r`. Unsloth and PEFT take `lora_alpha`. `mlx-lm` takes a `scale` value directly and errors if you pass `alpha`. Don't copy a config from one tool into another.

> ⚠ **QLoRA is not recommended for Qwen3.5.** Unsloth's Qwen3.5 page says it plainly: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense." On a GPU, use bf16 LoRA on a smaller model rather than QLoRA on a bigger one. On a Mac with `mlx-lm`, a 4-bit base is how the 4B fits at all. Treat that as an experiment and evaluate it harder.

<!-- section id="memory" toc="Will it fit?" kicker="Part 2 · Will it fit?" -->
## Will it fit on your machine?

**Start with the weights.** Weight storage ≈ parameters × bits ÷ 8. A 4B model at 16-bit is about 8 GB of weights; at 4-bit, about 2 GB.

<!-- widget:memory -->

**Training needs far more than the weights.** On top of them come activations saved for the backward pass, gradients, optimizer state, the adapter itself, and temporary buffers. Gradient checkpointing trades compute for memory. Shorter sequences and batch size 1 help most. No formula replaces measuring.

**Find your real budget.**
- **Mac:** MLX won't let the GPU use all of your unified memory. Ask it:
  ```bash
  python -c "import mlx.core as mx; print(mx.device_info())"
  ```
  Read `max_recommended_working_set_size`. On a 16 GB Apple Silicon Mac it measured **≈11.8 GiB**, about 75% of RAM.
- **NVIDIA:** run `nvidia-smi` and read the memory column. Close the browser and anything else using the GPU first.

**What fits, Qwen3.5 LoRA, October 2026:**

| Machine | Realistic target | Evidence |
|---|---|---|
| NVIDIA 8 GB | 0.8B or 2B | Unsloth bf16 LoRA: 0.8B ≈ 3 GB, 2B ≈ 5 GB |
| NVIDIA 12 GB | 4B, tight | Unsloth bf16 LoRA: 4B ≈ 10 GB |
| NVIDIA 16 GB | 4B, comfortable | same table |
| NVIDIA 24 GB | 9B, tight | Unsloth bf16 LoRA: 9B ≈ 22 GB |
| Mac 16 GB | 0.8B / 2B; 4B on a 4-bit base, tight | ~11.8 GiB GPU budget (measured) |
| Mac 24 GB | 4B on a 4-bit base | *Secondary:* 4B LoRA peaks of ~10.5–15 GB on 24 GB Macs |
| Mac 32 GB+ | 9B on a 4-bit base, plausible | *Secondary:* one 9B run peaked at 13.2 GB. *Untested here* |

> 📝 **Pick the smallest model that passes your test.** A well-tuned 2B that does one job reliably beats a 9B that doesn't fit. Run your test set against the *untrained* 0.8B, 2B and 4B first. If the 2B is close, tune the 2B.

<!-- section id="decide" toc="Decide before you train" kicker="Part 2 · Before you train" -->
## Decide before you train

Write these down before you touch a training command. They take an hour and save days.

1. **The behaviour, in one sentence.** "Turn a git diff into a one-line conventional-commit message." Not "be better at coding."
2. **A held-out test set.** 30–100 real inputs with the answer you'd accept, written *before* training and never shown to the model during training.
3. **A pass rule you can check.** Exact match, a JSON schema, a regex, a small script, or a rubric you apply by hand. "Looks good" is not a pass rule.
4. **The baseline score.** Run the untrained model on the test set with your best prompt. If it already passes, you're done. Don't train.
5. **A stop rule.** "If the adapter doesn't beat the baseline by X on the test set, I delete it."

> ⚠ **Thinking mode changes the baseline.** Qwen3.5 4B and 9B *think by default*: they write a `<think>` block before answering. Only the 0.8B and 2B default to non-thinking. Decide which mode you'll use in production and measure the baseline in that mode. Turn thinking off with `enable_thinking=False` in the chat template (Hugging Face, MLX, llama.cpp) or `--think=false` / a top-level `"think": false` in Ollama.

<!-- section id="data" toc="Build the dataset" kicker="Part 2 · The part that matters most" -->
## Build the dataset

**Format.** Both paths below read JSON Lines: one example per line. Use the chat format, so the model learns in the same shape it will be used in:
```json
{"messages": [{"role": "system", "content": "You write one-line commit messages."}, {"role": "user", "content": "diff --git a/api.py ..."}, {"role": "assistant", "content": "fix(api): handle empty payload"}]}
```

**Size.** For a narrow behaviour, 100–1,000 good examples is a normal starting range. Quality dominates: one wrong example teaches the wrong thing as hard as one right example teaches the right thing.

**Checklist for every dataset:**
- **Real inputs.** Use the messy inputs you actually see, not tidy ones you invent.
- **Consistent answers.** Two examples with the same kind of input should not disagree on format.
- **Coverage.** Include the edge cases and the "I can't do that" cases you want handled.
- **No leaks.** Remove duplicates, and make sure no test example, or a near-copy of one, is in the training data.
- **Splits.** `train.jsonl` for training, `valid.jsonl` (about 10%) to watch for overfitting during training, and your separate test set for the final decision.
- **Rights and privacy.** Only train on text you're allowed to use. Strip secrets, keys, emails and personal data. Assume anything in the training data can come back out of the model.

**Train only on the answers.** You usually want the model to learn the *assistant* turns, not to imitate your prompts. `mlx-lm` has `--mask-prompt`. Unsloth has a `train_on_responses_only` helper (see its docs for the exact call).

**Thinking traces.** If you train on plain answers with thinking off, expect the model's reasoning to weaken. Unsloth's Qwen3.5 guidance: if you want to keep reasoning, mix in reasoning-style examples, with "a minimum of 75% reasoning." If you only ever run the model with thinking off, train with thinking off and evaluate that way.

<!-- section id="mac" toc="Path A: Mac (mlx-lm)" kicker="Part 2 · Path A · Apple Silicon" -->
## Path A: a Mac with `mlx-lm`

`mlx-lm` is Apple's MLX library for language models. It's the training path with the fewest unknowns on a Mac in October 2026. It has supported the Qwen3.5 text model since v0.30.7 (Feb 2026). Its `LORA.md` family list still omits Qwen3/3.5; that list is stale, and the model code is there.

**1 — Install** (`mlx-lm` 0.32.0; needs macOS 15+ and Python ≥ 3.11):
```bash
uv venv --python 3.12 .venv-mlx && source .venv-mlx/bin/activate
uv pip install "mlx-lm[train]"
mlx_lm.lora --help
```

**2 — Pick a base.** For the 4B on a 16 GB Mac, use the 4-bit MLX conversion `mlx-community/Qwen3.5-4B-4bit` (≈3.0 GB). Pointing `--model` at a quantized model makes the run a QLoRA run. Its Hugging Face sidebar says "1.0B params"; that's a packed-weight counting quirk, not the wrong model. Vision weights are dropped on load, so this is text-only training. For a first run, use a smaller base such as the 0.8B or 2B.

**3 — Put the data in a folder:** `data/train.jsonl`, `data/valid.jsonl`, and `data/test.jsonl` for `--test`. `mlx-lm` auto-detects four formats: `chat`, `tools`, `completions`, `text`.

**4 — Train with conservative settings:**
```bash
mlx_lm.lora --model mlx-community/Qwen3.5-4B-4bit --train --data ./data \
  --batch-size 1 --num-layers 4 --grad-checkpoint \
  --max-seq-length 1024 --iters 600 --learning-rate 1e-5 --mask-prompt
```
Defaults worth knowing: rank 8, `scale` 20, Adam, learning rate 1e-5, adapters saved to `./adapters` every 100 steps. If you run out of memory, shorten `--max-seq-length` first, then lower `--num-layers`. Keep Activity Monitor open and watch memory pressure, not just the loss.

> ⚠ **Hybrid-architecture gotcha.** Qwen3.5 mixes two layer types: in every block of four, three are Gated DeltaNet (linear attention, `linear_attn`) and one is normal attention. By default `mlx-lm` adds LoRA to *every* linear layer in the last `--num-layers` blocks, DeltaNet included. A 2026 study on Qwen3.5-0.8B (arXiv 2604.22127) found adapting the DeltaNet backbone *destructive* (−14.8 points on GSM8K), while attention-only adaptation beat full adaptation with 5–10× fewer trainable parameters. To try attention-only, pass a YAML config (`-c config.yaml`) whose `lora_parameters.keys` lists only the attention projections. Check the exact module names with `model.named_modules()` first. *Untested here.*

**5 — Test and compare:**
```bash
mlx_lm.lora --model mlx-community/Qwen3.5-4B-4bit --adapter-path adapters --data ./data --test
mlx_lm.generate --model mlx-community/Qwen3.5-4B-4bit --adapter-path adapters --prompt "..."
```

**6 — Serve it.** `mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --adapter-path adapters` starts an OpenAI-compatible server on **127.0.0.1:8080**. Use 0.32.0 or later: 0.31.3 silently ignored `--adapter-path` (mlx-lm issue #1248, fixed by #1249 in Aug 2026).

> ◆ **No clean route into Ollama yet.** `mlx_lm.fuse --export-gguf` only supports llama, mistral and mixtral, and it refuses `qwen3_5`. A plausible path is `mlx_lm.fuse --dequantize` followed by `ollama create` from the fused safetensors folder (Ollama's importer registers `Qwen3_5ForCausalLM`). Whether `mlx-lm`'s fused layout imports cleanly is *untested*. Until you've proven it, serve with `mlx_lm.server`.

**Other trainers on a Mac:**
- **Unsloth** added an Apple Silicon (MLX) path in Aug 2026, for SFT and LoRA. GRPO and DPO aren't covered, and no Qwen3.5-on-Mac run has been published yet.
- **Axolotl** is "partially usable" on a Mac: no bitsandbytes, no flash-attn, no bf16.
- **LLaMA-Factory** ships Qwen3.5 templates but makes no Mac support claim.

<!-- section id="nvidia" toc="Path B: NVIDIA (Unsloth)" kicker="Part 2 · Path B · NVIDIA GPU" -->
## Path B: an NVIDIA PC with Unsloth

[Unsloth](https://unsloth.ai/docs) is the standard single-GPU fine-tuning library. It has a Qwen3.5 guide and exports straight to GGUF for Ollama and llama.cpp.

**1 — Check the GPU** before anything else:
```bash
nvidia-smi
```
Note the GPU memory and the CUDA version in the top-right corner. If the command fails, fix the NVIDIA driver first.

**2 — Install.** Pick one:
- **Linux or WSL** (uv):
  ```bash
  uv venv unsloth_env --python 3.13 && source unsloth_env/bin/activate
  uv pip install unsloth --torch-backend=auto
  ```
- **Native Windows** (Miniconda, PowerShell). Install PyTorch for *your* CUDA version first; the example uses CUDA 13.0, so `cu130`:
  ```bash
  conda create --name unsloth_env python==3.12 -y
  conda activate unsloth_env
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  python -c "import torch; print(torch.cuda.is_available())"
  pip install unsloth
  ```
  The PyTorch check must print `True`. Unsloth's Windows page calls Docker possibly "the easiest way for Windows users to get started" and suggests WSL if Docker doesn't see the GPU. GRPO and anything that needs vLLM require Linux or WSL.
- **No code:** Unsloth Studio is a browser UI for the same engine. On Windows, install it with `irm https://unsloth.ai/install.ps1 | iex`. On macOS, Linux or WSL, use `curl -fsSL https://unsloth.ai/install.sh | sh`.

Qwen3.5 needs `transformers` v5; older versions won't load it. Unsloth installs v5 by default.

**3 — Train.** This script is adapted from Unsloth's Qwen3.5 page: the model loading, LoRA settings and trainer arguments are theirs. The dataset-mapping lines are this guide's. Run it on 10 examples and `max_steps = 5` before a real run.
```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-4B",   # use "Qwen/Qwen3.5-2B" on an 8 GB card
    max_seq_length = 2048,
    load_in_4bit = False,             # QLoRA is not recommended for Qwen3.5
    load_in_16bit = True,
    full_finetuning = False,
)
model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 3407,
)

data = load_dataset("json", data_files = {"train": "data/train.jsonl"})["train"]
def to_text(example):   # the trainer wants a "text" column
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize = False, enable_thinking = False)}
data = data.map(to_text)

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = data,
    args = SFTConfig(
        max_seq_length = 2048, per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, warmup_steps = 10, max_steps = 100,
        logging_steps = 1, output_dir = "outputs_qwen35", optim = "adamw_8bit",
        seed = 3407, dataset_num_proc = 1,
    ),
)
trainer.train()
model.save_pretrained("finetuned_lora")
tokenizer.save_pretrained("finetuned_lora")
```
The first steps can be slow. Qwen3.5 uses custom Triton kernels that compile on first use. If you run out of memory, lower `max_seq_length`, keep batch size 1, and leave Unsloth's gradient checkpointing on.

**4 — Export for Ollama:**
```python
model.save_pretrained_gguf("gguf_out", tokenizer, quantization_method = "q4_k_m")   # or "q8_0", "f16"
```
Unsloth writes a `Modelfile` next to the GGUF. Point Ollama at it:
```bash
ollama create my-qwen -f <path-to-the-generated-Modelfile>
ollama run my-qwen
```
If the output is gibberish or repeats, the chat template or end-of-sequence token doesn't match what you trained with. Unsloth names this as the usual cause.

<!-- section id="eval" toc="Evaluate honestly" kicker="Part 2 · The only number that counts" -->
## Evaluate honestly

- **Same test, same settings.** Run the base model and the tuned model on your held-out test set with the same prompt, chat template, thinking mode and sampling settings.
- **Score with your pass rule,** not by eyeballing a few outputs. Write the numbers down.
- **Check what you might have broken.** Run 20–30 general prompts outside your task: a little arithmetic, a summary, a refusal case. Narrow training can make the model worse at everything else, and LoRA reduces this risk without removing it.
- **Watch for overfitting.** If validation loss rises while training loss keeps falling, the model is memorising. Use an earlier checkpoint, fewer steps, or more varied data.
- **Read the failures.** Most bad results trace back to inconsistent or wrong training examples. Fix the data, then retrain.
- **Keep the stop rule.** If the adapter doesn't beat the baseline by the margin you set before training, delete it and go back to prompting or retrieval.

<!-- section id="ship" toc="Serve and ship" kicker="Part 2 · Use it" -->
## Serve and ship

- **Keep the adapter separate** while you're still evaluating. Merge or convert only the version you'll actually use.
- **Mac:** `mlx_lm.server` serves the base plus adapter on `127.0.0.1:8080` with an OpenAI-compatible API.
- **PC:** `ollama create` from the GGUF and Modelfile, then use Ollama's API (`127.0.0.1:11434`, OpenAI-compatible at `/v1`).
- **Importing any GGUF into Ollama:** `FROM ./model.gguf` in a Modelfile, then `ollama create`. Ollama doesn't quantize GGUF files on import, so quantize first with `llama-quantize` or Unsloth's export.
- **Keep servers on loopback** (`127.0.0.1`). Neither server has authentication.
- **Sharing weights:** Qwen3.5 small models are Apache-2.0, which allows derivatives. Include the license, name the base model, and describe the training data honestly in the model card.

<!-- section id="beyond" toc="Beyond SFT" kicker="Part 2 · Next steps" -->
## Beyond supervised fine-tuning

- **Preference tuning (DPO) and reinforcement learning (GRPO)** teach the model to prefer some answers over others, or to optimise a score you compute. Unsloth supports both on NVIDIA GPUs. GRPO with vLLM needs Linux or WSL. Neither is covered by Unsloth's Mac path as of October 2026. Get plain SFT working and evaluated first.
- **Vision fine-tuning** (image + text) uses Unsloth's `FastVisionModel` on a GPU, or `mlx-vlm` on a Mac. `mlx-vlm`'s Qwen3.5 LoRA bugs were closed in September 2026. It needs much more memory than text-only training.
- **Abliteration** removes a model's refusal behaviour by editing weights directly, without training. It's a different technique with its own costs and responsibilities. Part 3 covers it.

<!-- section id="quiz" toc="Quiz" kicker="Part 2 · Check yourself" -->
## Can you spot the trap?

<!-- widget:quiz -->

<!-- section id="learn-ft" toc="Learn more" kicker="Part 2 · Fine-tune" -->
## Learn more: fine-tuning

- [Unsloth's fine-tuning guide for beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners) and its [Qwen3.5 page](https://unsloth.ai/docs/models/qwen3.5/fine-tune): the practical NVIDIA reference, with free Colab notebooks.
- [`mlx-lm` LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md): every flag for the Mac path, data formats, memory tips.
- [The Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1), especially the fine-tuning chapters, and the [TRL](https://huggingface.co/docs/trl/en/sft_trainer) and [PEFT](https://huggingface.co/docs/peft/index) docs that Unsloth builds on.
- [Maxime Labonne's LLM course](https://github.com/mlabonne/llm-course): a roadmap with notebooks, from fundamentals to fine-tuning and quantization.
- Papers: LoRA ([arXiv 2106.09685](https://arxiv.org/abs/2106.09685)), QLoRA ([arXiv 2305.14314](https://arxiv.org/abs/2305.14314)), and LoRA placement in hybrid models like Qwen3.5 ([arXiv 2604.22127](https://arxiv.org/abs/2604.22127)).
- [Qwen's training docs](https://qwen.readthedocs.io/en/latest/) list the frameworks the Qwen team supports: Axolotl, LLaMA-Factory, MS-SWIFT, Unsloth and verl.

<!-- part title="Part 3 · Abliteration" -->
<!-- section id="abl-what" toc="What abliteration is" kicker="Part 3 · Abliteration" -->
## What abliteration is

Instruction-tuned models refuse some requests. "Abliteration" is a technique that removes that refusal behaviour by editing the model's weights directly, without any training. This part explains how it works, what the research says it costs, how people do it on local hardware, and what you take on if you do. It's a conceptual and tooling-level map with no prompts and no recipes for harmful output.

**The mechanism.** Arditi et al. (NeurIPS 2024, [arXiv 2406.11717](https://arxiv.org/abs/2406.11717)) showed that in 13 open chat models up to 72B, refusal is mediated by **one direction** in the model's internal activations. Run a set of harmful and a set of harmless instructions through the model, record the activations at a chosen layer, and take the mean difference. That gives a unit vector `r`. Then **orthogonalize** every weight matrix that writes into the residual stream: `W' = W − r rᵀ W` for the embedding, attention-output and MLP-output weights. The model can no longer represent that direction, so it stops refusing. No gradient steps are involved.

**Why it's called that.** The word is a blend of "ablation" (removing a component to see what it does) and "obliteration." It started as interpretability research, a way to show that refusal is a simple, locatable feature, and became a tool.

**What it is not.** It isn't fine-tuning: there's no dataset of desired answers and nothing is learned. It isn't a jailbreak prompt: the change is permanent in the weights. And it isn't selective: it removes the model's refusal response as a whole, not only the refusals you consider mistaken.

<!-- section id="abl-research" toc="What research found" kicker="Part 3 · Abliteration" -->
## What later research found

- **It isn't always one direction.** Wollschläger et al. (ICML 2025, [arXiv 2502.17420](https://arxiv.org/abs/2502.17420)) found multiple independent refusal directions and multi-dimensional "concept cones," and showed that two directions being orthogonal doesn't mean they're independent when you intervene on them.
- **Refusals come in kinds.** Joad et al. ([arXiv 2602.02132](https://arxiv.org/abs/2602.02132), EMNLP 2026) separate safety refusals, capability-limit refusals ("I can't browse the web"), under-specification refusals and over-refusals, with typical cosine similarities between their directions of only 0.4 to 0.6. Removing one direction doesn't cleanly remove one kind.
- **Prompt design decides whether it works.** A 2026 study on Qwen3.5-2B ([arXiv 2603.22061](https://arxiv.org/abs/2603.22061)) found that *topic-matched* harmful and harmless prompt pairs produced no usable refusal direction at any layer tested, while unmatched pairs worked. The harmful and harmless sets need to differ in refusal and little else, which is harder than it sounds.
- **Gentler edits exist.** "Norm-preserving biprojected abliteration" ([grimjim, Nov 2025](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)) rescales weight rows after projection so their magnitudes don't change. "Gabliteration" ([arXiv 2512.18901](https://arxiv.org/abs/2512.18901), a single-author preprint) uses several SVD directions with ridge regularisation.
- **It can be defended against.** Abu Shairah et al. ([arXiv 2505.19056](https://arxiv.org/abs/2505.19056)) show that training a model to give a justification before refusing makes the refusal harder to locate as one direction: refusal rates dropped at most 10% under abliteration, versus 70–80% for undefended models. Expect future models to be harder to abliterate.

<!-- section id="abl-cost" toc="What it costs" kicker="Part 3 · Abliteration" -->
## What it costs the model

It is not free, and the cost varies a lot with the tool and settings.

- In Arditi et al.'s own evaluation, most benchmark scores stayed within confidence intervals, but **TruthfulQA dropped 1 to 3.5 points on every model** in their main table.
- A 2025 comparison of four abliteration tools on 16 instruction-tuned models of 7–14B ([arXiv 2512.13655](https://arxiv.org/abs/2512.13655)) measured GSM8K maths changes from **+1.5 to −18.8 points**, and KL divergence from the original from 0.04 to 1.6. Same idea, very different outcomes.
- Heretic's README shows the spread on one model: three abliterated Gemma-3-12B builds all reach 3 refusals in 100, with KL divergence from the original of **0.16, 0.45 and 1.04**. The lower number is the one that kept more of the model.

Two things follow. First, measure capability before and after, not just refusal rate. Second, prefer methods that optimise for low divergence from the original, because "stopped refusing" is easy and "stopped refusing without getting dumber" is the actual goal.

<!-- section id="abl-responsibility" toc="Responsibility and safety" kicker="Part 3 · Abliteration" -->
## Responsibility and safety

This section is the reason the rest of Part 3 exists. Read it before any of the routes.

**What you're removing.** A refusal direction doesn't separate "annoying over-refusal" from "genuinely protective refusal." The same edit that stops the model refusing to discuss a chemistry homework question also stops it refusing to help with things that hurt people. Published measurements of abliterated models show large drops in refusal across the board, not a targeted change. Whoever runs the result owns its behaviour.

**Who you're responsible to.**
- **Yourself.** Local, private use for research, writing or red-teaming your own systems is the common legitimate case. Even then, keep the model where only you can reach it.
- **Anyone who uses your deployment.** If you put an abliterated model behind an app, a bot or an API, you have removed a safety layer that users may assume is there. Add your own: input filtering, output review, a clear notice, and no access for minors.
- **Anyone you share weights with.** Publishing an abliterated model is legal under Apache-2.0, and people do. Name the base model, state what was changed and how, report the capability cost you measured, and don't describe it as "uncensored" as if that were a feature with no downside.
- **Third parties.** Laws on harmful content, defamation, harassment, fraud and export controls apply to what a model produces on your behalf. "The model said it" is not a defence anywhere.

**Rules this guide holds to:**
- **Never wire an abliterated model to tools with side effects.** No shell access, no file writes, no email or messaging, no browsing, no autonomous agent loops. A model with no refusal behaviour plus tools that act in the world is the single worst combination in this whole guide.
- **Keep it on loopback and out of shared memory.** Don't let an abliterated model write into a memory store or retrieval index that other models or people trust.
- **Prefer the smallest change that solves your real problem.** If the actual issue is a model refusing a benign task, try a system prompt, a different model, or a few examples first. Most "the model won't help me" problems are solved that way.
- **Measure capability loss and write it down.** Section "Evaluate" below.
- **Don't do it to models you don't have the right to modify,** and don't redistribute weights whose license forbids it.

**A short decision check.** Before you abliterate, answer honestly:
1. What specific benign task is the base model refusing, and have I tried a prompt or a different model?
2. Who else will ever be able to reach this model, and what protects them?
3. Will it ever touch tools, memory or the network?
4. Will I measure and record what it lost?
5. If the output were shown next to my name, would I defend it?

If any answer makes you uncomfortable, stop there.

**Why this section isn't a refusal to explain.** The research is public and peer-reviewed, the tools are open source, and the ready-made models have hundreds of thousands of downloads. Someone who wants to do this will. The useful thing a guide can do is make the costs, the responsibilities and the safer choices clear, and point at the real literature rather than at folklore.

<!-- section id="abl-routes" toc="How people do it locally" kicker="Part 3 · Abliteration" -->
## How people do it locally

**Route A: use a published model.** The lowest-effort path, and the one most people take. A widely pulled Qwen3.5 build is `huihui_ai/qwen3.5-abliterated` on Ollama (about 678K pulls):
```bash
ollama run huihui_ai/qwen3.5-abliterated:4b    # 3.3–3.4 GB
ollama run huihui_ai/qwen3.5-abliterated:9b    # 6.6 GB
```
Always write a tag. Untagged `latest` is the **27B (17 GB)** and won't load on a 16 GB machine. The repo also carries `-Claude`, `-Qwopus` and `-OmniCoder` tags, which are *different fine-tunes*, not plain abliterations. The publisher's model card says the 4B was also fine-tuned with TRL on non-thinking data, text only, with the multi-token-prediction weights removed, so expect it to differ from the base in more than refusals. These are third-party weights; everything in Part 1's "Keep it on your machine" applies, and so does the whole responsibility section above.

**Route B: run Heretic yourself.** [`p-e-w/heretic`](https://github.com/p-e-w/heretic) (AGPL-3.0, `pip install -U heretic-llm`, version 1.4.0 in October 2026) automates the direction search with Optuna. It optimises refusal count *and* KL divergence from the original together, which is why its builds tend to keep more capability. Two facts matter for a local machine: **Qwen3.5's hybrid layers have been supported since v1.3.0**, and **basic Apple Silicon (MPS) support has existed since v1.1.0**, tested on a Qwen2.5-0.5B on an M4 Max with 128 GB.
- On an NVIDIA GPU, the README's reference point is 20–30 minutes for a Qwen3-4B on an RTX 3090 (24 GB).
- On a 16 GB Mac, the **0.8B and 2B** are the realistic experiment. The **4B** in bf16 is 9.32 GB of weights, under the ~11.8 GiB GPU budget on paper but *untested* and likely tight once activations and trials are added. The **9B** (19.3 GB bf16) does not fit.
- Version 1.4.0 pins `bitsandbytes~=0.49`, whose 4-bit path on Apple Silicon is the slow one. Don't count on quantized loading to rescue a bigger model on a Mac.

Heretic writes Hugging Face safetensors. To run the result in Ollama, convert with llama.cpp's `convert_hf_to_gguf.py` (it registers `Qwen3_5ForCausalLM`), quantize with `llama-quantize`, then `ollama create` from the GGUF.

**Route C: do it by hand.** The 2024-era reference scripts (FailSpy's `abliterator`, Sumandora's `remove-refusals-with-transformers`, mlabonne's notebook) assume CUDA and a Llama-style layout with `self_attn.o_proj` in every block. Qwen3.5-4B and 9B have **24 Gated DeltaNet layers and 8 full-attention layers**; the attention output projection is `linear_attn.out_proj` on the former and `self_attn.o_proj` on the latter, alongside `mlp.down_proj`. Old code breaks on this. Read Arditi's paper and Heretic's Qwen3.5 pull request (#187) before adapting anything, and treat it as a research project, not a weekend task.

**OBLITERATUS.** [`elder-plinius/OBLITERATUS`](https://github.com/elder-plinius/OBLITERATUS) (AGPL-3.0) is a larger toolkit with several extraction methods (mean difference, PCA, sparse autoencoders, whitened SVD) and an MLX backend. Its README lists no validated models, and the Qwen work documented in its research notes is a Qwen3.8-27B run on a CUDA device, so it isn't a tested Mac path for the small models. Note that telemetry is **on by default on its hosted Space** and opt-in locally.

<!-- section id="abl-eval" toc="Evaluate" kicker="Part 3 · Abliteration" -->
## Evaluate any abliterated model

Measure three things, before and after, with the same prompts and settings:
1. **Refusal rate** on a held-out set of prompts you did *not* use to find the direction. Include benign prompts that the base model wrongly refused, since those are the ones you presumably cared about.
2. **Divergence from the original**: KL divergence or perplexity on ordinary text. Heretic reports KL; for a published model, compare answers on 30 neutral prompts side by side with the base.
3. **A capability set you care about**: a little maths, a summary, a structured-output task, your own real prompts.

Run everything with thinking **on and off**: the 4B and 9B think by default, and different tools read different token positions. Write the three numbers in the model card or your notes. A model that stops refusing but loses fifteen points of maths has cost you more than it gained.

<!-- section id="learn-abl" toc="Learn more" kicker="Part 3 · Abliteration" -->
## Learn more: abliteration and interpretability

Start with the research, then the tools.

- **The paper:** Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" ([arXiv 2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024). The authors' [LessWrong write-up](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) is the readable version.
- **The walkthrough:** Maxime Labonne, ["Uncensor any LLM with abliteration"](https://huggingface.co/blog/mlabonne/abliteration) (2024). Clear code and figures, Llama-era layer names.
- **The follow-ups:** [Wollschläger et al. 2025](https://arxiv.org/abs/2502.17420) on concept cones; [Joad et al. 2026](https://arxiv.org/abs/2602.02132) on kinds of refusal; [Petrov 2026](https://arxiv.org/abs/2603.22061) on Qwen3.5 and prompt matching; [Young 2025](https://arxiv.org/abs/2512.13655) comparing four tools; [grimjim's norm-preserving method](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration); [Abu Shairah et al. 2025](https://arxiv.org/abs/2505.19056) on defending against it.
- **The tools:** [Heretic](https://github.com/p-e-w/heretic) and [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS), both with READMEs that document their methods and measured results.
- **The field behind it:** abliteration is applied mechanistic interpretability. Neel Nanda's [getting-started guide](https://www.neelnanda.io/mechanistic-interpretability/getting-started) and Anthropic's [Transformer Circuits](https://transformer-circuits.pub/) publications are where the ideas about directions, features and circuits come from.
- **Published models:** the [huihui_ai Qwen3.5 builds](https://ollama.com/huihui_ai/qwen3.5-abliterated) on Ollama, with the caveats in Route A.

<!-- part title="Reference" -->
<!-- section id="trouble" toc="Troubleshooting" kicker="Reference" -->
## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ollama run qwen3.5` downloads 7 GB | `latest` is the 9B | Name the tag: `qwen3.5:4b` |
| Model runs on CPU, very slow | GPU not detected: old driver, unsupported card, or no ROCm/CUDA | Check `ollama ps` and the GPU requirements; update the driver |
| Machine swaps or freezes while a model loads | Model plus KV cache exceeds free memory | Smaller model or quant, shorter `num_ctx`, one model at a time |
| A server on the network can reach my Ollama | Bound to `0.0.0.0` | Set `OLLAMA_HOST=127.0.0.1:11434` and restart |
| Abliterated model is coherent but worse at maths | Capability loss from the edit | Try a lower-divergence build or method; measure before trusting it |
| Out of memory | Sequences too long, too many layers, or the model is too big | Shorten max sequence length, batch size 1, gradient checkpointing on, then a smaller model |
| Loss doesn't move | Learning rate too low, or the prompt is masked wrongly | Check that the assistant turns are present; try a slightly higher LR on a 20-step run |
| Loss falls, test score doesn't | Overfitting, or the test differs from the training data | More varied data, fewer steps, use an earlier checkpoint |
| Gibberish or repetition after export | Chat template or end-of-sequence token mismatch | Serve with the same template you trained with |
| `mlx_lm.server` ignores the adapter | `mlx-lm` 0.31.3 bug | Upgrade to 0.32.0+ |
| `--export-gguf` error on Qwen3.5 | `mlx_lm.fuse` only exports llama/mistral/mixtral | Serve with `mlx_lm.server` |
| Qwen3.5 won't load in Python | `transformers` older than v5 | Upgrade `transformers` |
| First training steps very slow | Triton kernels compiling | Wait; later steps are faster |
| Answers start with a long `<think>` block | Thinking mode is on by default for 4B/9B | Pass `enable_thinking=False`, or train and serve consistently with thinking on |

<!-- section id="sources" toc="Sources" kicker="Reference" -->
## Sources

Every version, number and command in this guide traces to one of these, checked in October 2026.

- Running: [Ollama docs](https://docs.ollama.com/) (quickstart, GPU, import), [Ollama Qwen3.5 tags](https://ollama.com/library/qwen3.5/tags), [Hugging Face `hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), [Hub + Ollama](https://huggingface.co/docs/hub/en/ollama), [Qwen docs](https://qwen.readthedocs.io/en/latest/), [QwenLM release notes](https://github.com/QwenLM/Qwen3.8)
- `mlx-lm`: [PyPI](https://pypi.org/project/mlx-lm/), [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [qwen3_5 model code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py), [fuse.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py)
- Unsloth: [Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [install](https://unsloth.ai/docs/get-started/install), [Windows install](https://unsloth.ai/docs/get-started/install/windows-installation), [saving to Ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama)
- Qwen3.5 model cards: [4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Papers: LoRA [arXiv 2106.09685](https://arxiv.org/abs/2106.09685), QLoRA [arXiv 2305.14314](https://arxiv.org/abs/2305.14314), LoRA placement in hybrid models [arXiv 2604.22127](https://arxiv.org/abs/2604.22127)
- Ollama: [importing models](https://docs.ollama.com/import), [thinking](https://ollama.com/blog/thinking)
- Axolotl on Mac: [docs](https://docs.axolotl.ai/docs/mac.html)
- Abliteration: [Arditi et al.](https://arxiv.org/abs/2406.11717), [Wollschläger et al.](https://arxiv.org/abs/2502.17420), [Joad et al.](https://arxiv.org/abs/2602.02132), [Petrov](https://arxiv.org/abs/2603.22061), [Gülmez](https://arxiv.org/abs/2512.18901), [Young](https://arxiv.org/abs/2512.13655), [Abu Shairah et al.](https://arxiv.org/abs/2505.19056), [grimjim](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration), [Heretic](https://github.com/p-e-w/heretic), [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS), [huihui_ai on Ollama](https://ollama.com/huihui_ai/qwen3.5-abliterated)

<!-- section id="caveats" toc="Caveats" kicker="Reference" -->
## Caveats

- **Nothing here was benchmarked on one fixed machine.** The memory figures come from vendor tables, one measured MLX budget on a 16 GB Mac, and labelled secondary reports. Measure your own peak.
- **This field moves monthly.** `mlx-lm`, Unsloth, `transformers` and Ollama all changed Qwen3.5 support during 2026. Re-check versions before following a command.
- **Untested paths are labelled.** Attention-only LoRA keys in `mlx-lm`, the fused-safetensors-to-Ollama import, and Unsloth's Mac path for Qwen3.5 are not proven here.
- **The landscape tables are a snapshot, dated 2026-10-15.** Model families, sizes and licenses change weekly. Every row was checked against a model card or Ollama's library page that day; the watch list under the tables names what wasn't. Re-check the model card before you commit to a family.
- **Part 3 is deliberately conceptual.** It explains the research and names the tools. It does not provide prompt sets, harmful examples or step-by-step editing code.
- **A tuned model is still a language model.** It can be confidently wrong. Keep validators and human review wherever mistakes cost something.

