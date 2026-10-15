# Fine-tune Qwen locally: a first-principles guide for Mac and PC

> Written **October 2026** for Qwen3.5 (0.8B / 2B / 4B / 9B). Every command, version and number was checked against a primary source that month: PyPI, the `ml-explore/mlx-lm` and Unsloth docs, Hugging Face model cards, arXiv. Anything else is labelled *secondary* (blog or forum) or *untested*. Fast-moving facts go stale. Re-check versions before you rely on them.

**Who this is for:** you can run a local model already (if not, start with the companion [Qwen Local guide](https://github.com/Joona-t/qwen-local-guide)) and you want the model to *behave* differently: a fixed output format, a house style, a narrow task it keeps getting wrong. You have either an Apple Silicon Mac or a PC with an NVIDIA GPU. No paid API is involved anywhere.

**The three rules this guide keeps coming back to:**
1. **Most problems don't need fine-tuning.** Try a better prompt, a few examples, retrieval or a better tool first. Fine-tune only when you can show the base model failing on a test set.
2. **Small models train on small machines.** A 0.8B or 2B LoRA fits almost anywhere. A 4B fits with care. A 9B needs 24 GB-class hardware.
3. **Evaluate or it didn't happen.** Training loss going down proves the model fits *your training examples*. Only a held-out test, written before you train, tells you whether it got better.

<!-- section id="basics" toc="1 · What changes" kicker="Part 1 · Foundations" -->
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

<!-- section id="step" toc="2 · One learning step" kicker="Part 2 · Follow the numbers" -->
## One learning step, made visible

Supervised fine-tuning (SFT) shows the model examples of the answers you want. A **forward pass** predicts. A **loss** measures how wrong the prediction was. **Backpropagation** computes how each weight affected that loss. An **optimizer** moves the weights a little in the direction that lowers it. Repeat a few thousand times.

The toy below is one weight, one example and one step at a time. It isn't Qwen, but the arithmetic is the same arithmetic.

<!-- widget:gradient -->

**Loss is feedback, not truth.** Cross-entropy loss punishes the model for giving low probability to the *target*. If the target is wrong, a lower loss means the model learned the wrong thing better. Flip the toy's label to "wrong" and watch the loss fall anyway. That's why the quality of your examples matters more than their quantity.

**The learning rate is the step size.** Too small and nothing changes. Too large and the model overshoots, then forgets what it knew. One *secondary* report of a Qwen3.5 LoRA on a Mac saw a learning rate of 1e-4 *triple* the error count against the untouched base, while 1e-5 cut it by two thirds. Start small.

<!-- section id="lora" toc="3 · LoRA & QLoRA" kicker="Part 3 · Change fewer numbers" -->
## Full fine-tuning, LoRA and QLoRA

**Full fine-tuning** updates every weight. It's flexible, but you need memory for the weights, their gradients and the optimizer state for *all* of them. Unsloth puts it at roughly 4× the memory of LoRA.

**LoRA** (Hu et al., 2021, arXiv 2106.09685) freezes the original weights `W` and trains two small matrices per targeted layer: `A` (r × d_in) and `B` (d_out × r). Their product `B·A` has the same shape as `W` and gets added to it, scaled. The rank `r` is the bottleneck: at `r = 16` on a 4096 × 4096 layer you train 131,072 numbers instead of 16.7 million. The adapter is a separate small file, so the base model stays untouched and you can switch adapters on and off.

**QLoRA** (Dettmers et al., 2023, arXiv 2305.14314) is LoRA on top of a base model stored in 4-bit. The base is smaller in memory, and the adapters still train in higher precision.

<!-- widget:lora -->

> ◆ **Scale conventions differ between tools.** The standard formula scales the update by `alpha / r`. Unsloth and PEFT take `lora_alpha`. `mlx-lm` takes a `scale` value directly and errors if you pass `alpha`. Don't copy a config from one tool into another.

> ⚠ **QLoRA is not recommended for Qwen3.5.** Unsloth's Qwen3.5 page says it plainly: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense." On a GPU, use bf16 LoRA on a smaller model rather than QLoRA on a bigger one. On a Mac with `mlx-lm`, a 4-bit base is how the 4B fits at all. Treat that as an experiment and evaluate it harder.

<!-- section id="memory" toc="4 · Will it fit?" kicker="Part 4 · Will it fit?" -->
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

<!-- section id="decide" toc="5 · Decide first" kicker="Part 5 · Before you train" -->
## Decide before you train

Write these down before you touch a training command. They take an hour and save days.

1. **The behaviour, in one sentence.** "Turn a git diff into a one-line conventional-commit message." Not "be better at coding."
2. **A held-out test set.** 30–100 real inputs with the answer you'd accept, written *before* training and never shown to the model during training.
3. **A pass rule you can check.** Exact match, a JSON schema, a regex, a small script, or a rubric you apply by hand. "Looks good" is not a pass rule.
4. **The baseline score.** Run the untrained model on the test set with your best prompt. If it already passes, you're done. Don't train.
5. **A stop rule.** "If the adapter doesn't beat the baseline by X on the test set, I delete it."

> ⚠ **Thinking mode changes the baseline.** Qwen3.5 4B and 9B *think by default*: they write a `<think>` block before answering. Only the 0.8B and 2B default to non-thinking. Decide which mode you'll use in production and measure the baseline in that mode. Turn thinking off with `enable_thinking=False` in the chat template (Hugging Face, MLX, llama.cpp) or `--think=false` / a top-level `"think": false` in Ollama.

<!-- section id="data" toc="6 · Build the dataset" kicker="Part 6 · The part that matters most" -->
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

<!-- section id="mac" toc="A · Mac (mlx-lm)" kicker="Path A · Apple Silicon" -->
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

**6 — Serve it.** `mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --adapter-path adapters` starts an OpenAI-compatible server on **127.0.0.1:8080**. Use 0.32.0 or later: 0.31.3 silently ignored `--adapter-path`.

> ◆ **No clean route into Ollama yet.** `mlx_lm.fuse --export-gguf` only supports llama, mistral and mixtral, and it refuses `qwen3_5`. A plausible path is `mlx_lm.fuse --dequantize` followed by `ollama create` from the fused safetensors folder (Ollama's importer registers `Qwen3_5ForCausalLM`). Whether `mlx-lm`'s fused layout imports cleanly is *untested*. Until you've proven it, serve with `mlx_lm.server`.

**Other trainers on a Mac:**
- **Unsloth** added an Apple Silicon (MLX) path in Aug 2026, for SFT and LoRA. GRPO and DPO aren't covered, and no Qwen3.5-on-Mac run has been published yet.
- **Axolotl** is "partially usable" on a Mac: no bitsandbytes, no flash-attn, no bf16.
- **LLaMA-Factory** ships Qwen3.5 templates but makes no Mac support claim.

<!-- section id="nvidia" toc="B · NVIDIA (Unsloth)" kicker="Path B · NVIDIA GPU" -->
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

<!-- section id="eval" toc="7 · Evaluate" kicker="Part 7 · The only number that counts" -->
## Evaluate honestly

- **Same test, same settings.** Run the base model and the tuned model on your held-out test set with the same prompt, chat template, thinking mode and sampling settings.
- **Score with your pass rule,** not by eyeballing a few outputs. Write the numbers down.
- **Check what you might have broken.** Run 20–30 general prompts outside your task: a little arithmetic, a summary, a refusal case. Narrow training can make the model worse at everything else, and LoRA reduces this risk without removing it.
- **Watch for overfitting.** If validation loss rises while training loss keeps falling, the model is memorising. Use an earlier checkpoint, fewer steps, or more varied data.
- **Read the failures.** Most bad results trace back to inconsistent or wrong training examples. Fix the data, then retrain.
- **Keep the stop rule.** If the adapter doesn't beat the baseline by the margin you set before training, delete it and go back to prompting or retrieval.

<!-- section id="ship" toc="8 · Serve & ship" kicker="Part 8 · Use it" -->
## Serve and ship

- **Keep the adapter separate** while you're still evaluating. Merge or convert only the version you'll actually use.
- **Mac:** `mlx_lm.server` serves the base plus adapter on `127.0.0.1:8080` with an OpenAI-compatible API.
- **PC:** `ollama create` from the GGUF and Modelfile, then use Ollama's API (`127.0.0.1:11434`, OpenAI-compatible at `/v1`).
- **Importing any GGUF into Ollama:** `FROM ./model.gguf` in a Modelfile, then `ollama create`. Ollama doesn't quantize GGUF files on import, so quantize first with `llama-quantize` or Unsloth's export.
- **Keep servers on loopback** (`127.0.0.1`). Neither server has authentication.
- **Sharing weights:** Qwen3.5 small models are Apache-2.0, which allows derivatives. Include the license, name the base model, and describe the training data honestly in the model card.

<!-- section id="beyond" toc="9 · Beyond SFT" kicker="Part 9 · Next steps" -->
## Beyond supervised fine-tuning

- **Preference tuning (DPO) and reinforcement learning (GRPO)** teach the model to prefer some answers over others, or to optimise a score you compute. Unsloth supports both on NVIDIA GPUs. GRPO with vLLM needs Linux or WSL. Neither is covered by Unsloth's Mac path as of October 2026. Get plain SFT working and evaluated first.
- **Vision fine-tuning** (image + text) uses Unsloth's `FastVisionModel` on a GPU, or `mlx-vlm` on a Mac. `mlx-vlm`'s Qwen3.5 LoRA bugs were closed in September 2026. It needs much more memory than text-only training.
- **Abliteration** removes a model's refusal behaviour by editing weights directly, without training. It's a different technique with its own costs and responsibilities. See the [abliteration appendix](https://github.com/Joona-t/qwen-local-guide/blob/main/source/qwen-local-guide.md#appendix--abliteration-what-it-is-what-it-costs-how-people-do-it-locally) in the companion guide.

<!-- section id="quiz" toc="Quiz" kicker="Check yourself" -->
## Can you spot the trap?

<!-- widget:quiz -->

<!-- section id="trouble" toc="Troubleshooting" kicker="When it breaks" -->
## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Out of memory | Sequences too long, too many layers, or the model is too big | Shorten max sequence length, batch size 1, gradient checkpointing on, then a smaller model |
| Loss doesn't move | Learning rate too low, or the prompt is masked wrongly | Check that the assistant turns are present; try a slightly higher LR on a 20-step run |
| Loss falls, test score doesn't | Overfitting, or the test differs from the training data | More varied data, fewer steps, use an earlier checkpoint |
| Gibberish or repetition after export | Chat template or end-of-sequence token mismatch | Serve with the same template you trained with |
| `mlx_lm.server` ignores the adapter | `mlx-lm` 0.31.3 bug | Upgrade to 0.32.0+ |
| `--export-gguf` error on Qwen3.5 | `mlx_lm.fuse` only exports llama/mistral/mixtral | Serve with `mlx_lm.server` |
| Qwen3.5 won't load in Python | `transformers` older than v5 | Upgrade `transformers` |
| First training steps very slow | Triton kernels compiling | Wait; later steps are faster |
| Answers start with a long `<think>` block | Thinking mode is on by default for 4B/9B | Pass `enable_thinking=False`, or train and serve consistently with thinking on |

<!-- section id="sources" toc="Sources" kicker="Evidence" -->
## Sources

- `mlx-lm`: [PyPI](https://pypi.org/project/mlx-lm/), [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [qwen3_5 model code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py), [fuse.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py)
- Unsloth: [Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [install](https://unsloth.ai/docs/get-started/install), [Windows install](https://unsloth.ai/docs/get-started/install/windows-installation), [saving to Ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama)
- Qwen3.5 model cards: [4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Papers: LoRA [arXiv 2106.09685](https://arxiv.org/abs/2106.09685), QLoRA [arXiv 2305.14314](https://arxiv.org/abs/2305.14314), LoRA placement in hybrid models [arXiv 2604.22127](https://arxiv.org/abs/2604.22127)
- Ollama: [importing models](https://docs.ollama.com/import), [thinking](https://ollama.com/blog/thinking)
- Axolotl on Mac: [docs](https://docs.axolotl.ai/docs/mac.html)

<!-- section id="caveats" toc="Caveats" kicker="Honesty" -->
## Caveats

- **Nothing here was benchmarked on one fixed machine.** The memory figures come from vendor tables, one measured MLX budget on a 16 GB Mac, and labelled secondary reports. Measure your own peak.
- **This field moves monthly.** `mlx-lm`, Unsloth, `transformers` and Ollama all changed Qwen3.5 support during 2026. Re-check versions before following a command.
- **Untested paths are labelled.** Attention-only LoRA keys in `mlx-lm`, the fused-safetensors-to-Ollama import, and Unsloth's Mac path for Qwen3.5 are not proven here.
- **A tuned model is still a language model.** It can be confidently wrong. Keep validators and human review wherever mistakes cost something.
