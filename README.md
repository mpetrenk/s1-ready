# System 1 Ready (`s1-ready`)

**Turn a plain-English decision and a pile of unlabeled text into a clean,
calibrated, training-ready dataset for a fast decision model.**

> **What "S1" means.** "S1" is **System 1** — the fast, intuitive, single-pass
> judgment from Kahneman's two-system model of thinking (System 1 = snap
> decision, System 2 = slow deliberation). A *System 1 model* is one you ask a
> question and it answers in one shot: route this ticket, flag this message,
> score this risk. `s1-ready` produces the data that makes a model like that
> good. It is itself the System 2 work — the slow, deliberate labeling,
> decontamination, and balancing — that earns a fast model.

### What it's for

`s1-ready` exists to **accelerate decision-model development** — to produce the
quality datasets that fast "System 1" decision models need to get good. A
decision model takes a state (text, email, ticket, JSON) plus a set of typed
questions and returns a structured answer — a choice, a score, a yes/no — in a
single pass, instead of generating free text. That category now has real
implementations:

- **[Jev](https://www.anth.us/blog/fine-tuning-jev/)** (TypeSafe AI) named the
  category — a hosted, closed typed-decision model. You *can't* fine-tune Jev;
  it's an API and everyone shares the same weights.
- **[Laya](https://github.com/NandhaKishorM/laya)** (Convai Innovations) is the
  open-source counterpart — Apache-2.0, open weights, `pip install laya`. It is
  built to be fine-tuned, and its own docs are blunt about it: *fine-tuning is
  where most of the value is*. The shipped checkpoints are near chance on a
  domain's typed decisions zero-shot and jump sharply once fine-tuned on
  in-domain data.
- Or **roll your own** small decision classifier with
  **[Unsloth](https://github.com/unslothai/unsloth)**, TRL, or Axolotl.

Every one of those paths has the same prerequisite and the same bottleneck:
**good labeled option data**. Training a decision model is cheap and nearly
automatic now; assembling a clean, decontaminated, balanced dataset of `{state,
questions, gold}` rows is the hard part. That is exactly what `s1-ready`
produces — the System 2 prep work (slow, deliberate labeling) that earns a fast
System 1 model. Point it at a Laya fine-tune, an Unsloth LoRA, or any SFT
trainer and the dataset drops straight in.

```python
from s1_ready import SchemaInferencer, DataForge, Labeler

spec  = SchemaInferencer.from_prompt("Flag whether an SMS message is spam")
forge = DataForge(spec).ingest("messages.txt")
rows  = Labeler(teacher=["openai:gpt-4o-mini", "anthropic:claude-haiku-5-5"]).label(forge)
rows.decontaminate(test_set).balance().to_jsonl("train.jsonl")   # feed straight to Unsloth
```

Describe the decision, point it at your corpus, get a `train.jsonl` you can hand
straight to Unsloth or any SFT trainer.

## Proven end-to-end

A one-sentence decision + a public SMS corpus → a dataset → a fine-tuned
classifier, **trained with [Unsloth](https://github.com/unslothai/unsloth) on a
free Colab T4**. On a held-out, decontaminated test set the s1-ready-trained
model reaches **F1 0.965**, versus **0.297** for the same base model untrained —
the generated dataset is the entire difference.

![SMS spam benchmark](examples/case_study/metrics_comparison.png)

| model | accuracy | precision | recall | F1 |
|---|---:|---:|---:|---:|
| **Llama-3.1 fine-tuned (s1-ready)** | **0.991** | **0.961** | **0.969** | **0.965** |
| Llama-3.1 base (no fine-tune) | 0.425 | 0.175 | 0.977 | 0.297 |
| TF-IDF + Naive Bayes | 0.970 | 0.839 | 0.938 | 0.886 |

The fine-tune uses **Unsloth** (4-bit LoRA on Llama-3.1-8B) so the whole run
fits a free Colab T4. Full method, charts, and an honest read of when this route
is worth it (vs cheap classical ML):
**[examples/case_study/CASE_STUDY_sms_spam.md](examples/case_study/CASE_STUDY_sms_spam.md)**.
Reproduce it one-click in Colab: **[`s1_spam_colab.ipynb`](examples/case_study/s1_spam_colab.ipynb)**.

## Scope

**`s1-ready` is built to work with any decision-based use case.** If your task
can be framed as a decision over an input, the pipeline is use-case-agnostic:
you change the use-case sentence and the corpus, nothing else. A "decision" here
is one of three shapes:

- **choice** — pick exactly one option from a fixed set (routing, triage, tagging, intent)
- **bool** — a yes/no flag (is-refund, is-urgent, is-spam)
- **score** — a bounded ordinal/numeric rating (priority 1–5, severity, risk)

Routing, triage, classification, flagging, severity rating, and intent
detection all fit. The same pipeline handles each by swapping `--use-case`.

What it is **not** built for, so the claim stays honest:

- **Open-ended generation** (summaries, free-text answers, rewriting, span
  extraction) — there is no single gold answer to vote on, so the consensus /
  agreement machinery doesn't apply.
- **Multi-step reasoning or tool-use traces** — this prepares System 1
  (single-pass judgment) data, not System 2 chains.
- **Non-text inputs** — states are strings.

> Two honest qualifiers within scope: (1) schema inference is heuristic — it
> *proposes* the question + options from your sentence, and you should review or
> hand-write the option set for anything serious; (2) "decision-based" means the
> data has a decision attached — raw narrative text isn't a use case until you
> define what to decide about it. A human should confirm the schema for anything
> regulated.

## Install

> **Not on PyPI yet.** Install from source until the first release is published.

```bash
pip install "git+https://github.com/mpetrenk/s1-ready.git"          # core, zero deps
pip install "s1-ready[llm] @ git+https://github.com/mpetrenk/s1-ready.git"  # + LLM teachers
# or, from a local clone:
git clone https://github.com/mpetrenk/s1-ready.git
pip install -e "s1-ready[dev]"   # editable, with test deps
```

Once a release is on PyPI this becomes the usual `pip install s1-ready` /
`pip install "s1-ready[llm]"`.

The core pipeline has **zero runtime dependencies** — it runs on the stdlib with
a built-in heuristic teacher, so you can try it with no API keys. The `[llm]`
extra adds the OpenAI / Anthropic / Bedrock teachers; `[dev]` adds `pytest`.

## Quickstart (library)

```python
from s1_ready import SchemaInferencer, DataForge, Labeler

# 1. Describe the use case -> infer the question schema (choice/bool/score + options)
spec = SchemaInferencer.from_prompt(
    "Route support tickets to a team and flag refunds",
    sample_states=open("tickets.txt"),
)

# 2. Build + label the corpus
forge = DataForge(spec)
forge.ingest("tickets.txt")
forge.synthesize(n=500, balance=True)
rows = Labeler(teacher=["bedrock:claude", "gpt"]).label(forge, vote="majority")

# 3. Quality gates -> export System 1-ready JSONL
ds = rows.decontaminate(test_set).balance().confidence_filter(min=0.7)
ds.to_jsonl("train.jsonl")   # {state, questions, gold}
```

## Quickstart (CLI)

```bash
s1-ready prepare \
  --use-case "Route support tickets to a team and flag refunds" \
  --corpus tickets.txt \
  --synthesize 500 \
  --teacher heuristic \
  --out train.jsonl
```

(`s1ready` is a shorter alias for the same command.)

## Pipeline

| Stage          | Module       | What it automates                                             |
|----------------|--------------|---------------------------------------------------------------|
| Schema infer   | `schema.py`  | sentence -> (state, question type, option set)                |
| Forge          | `forge.py`   | ingest corpus, synthesize fillers, balance option skew        |
| Label          | `labeler.py` | multi-teacher consensus vote + agreement score + review queue |
| Gates          | `gates.py`   | decontaminate train/test, split, confidence filter            |
| Export         | `export.py`  | -> typed-decisions / unsloth / conversational / raw JSONL      |

The agreement score and review queue are first-class: any low-agreement row is
flagged for human spot-check. For regulated use cases (fraud, clinical), route
the review queue to a human before export.

### Schema inference, validated

Schema inference (sentence → questions + options) is the step most tools get
wrong, so it has a committed **100-case benchmark**
([`examples/schema_benchmark/`](examples/schema_benchmark/)) spanning
choice/bool/score, single- and multi-question use cases, varied domains and
phrasings. Measured accuracy:

| path | score | notes |
|------|------:|-------|
| heuristic (default, zero-dep) | **100 / 100** | regex + keyword rules, tuned against this set |
| LLM teacher (`openai:gpt-4o-mini`) | **96 / 100** | extraction prompt; of the 4 misses, 2 are defensible bool-vs-2-option choices, 1 is cosmetic (`feature_request` vs `feature request`), 1 is a real truncation — ~99 on a fair read |

The benchmark runs as a `pytest` regression gate, so neither path can silently
degrade. Reproduce:

```bash
python examples/schema_benchmark/run_benchmark.py                       # heuristic
S1_BENCH_TEACHER="openai:gpt-4o-mini" python examples/schema_benchmark/run_benchmark.py  # LLM
```

Honest caveat: 100/100 for the heuristic partly reflects that the gold was
written against its expected behavior; the LLM path was scored cold and
generalizes better to phrasings outside these 100 patterns. For anything
high-stakes, review the inferred schema (or hand-write it) before labeling.

## Export formats

Pick with `--format` (CLI) or `fmt=` (library). All are newline-delimited JSON.

| `--format`        | Shape                                             | For                                  |
|-------------------|---------------------------------------------------|--------------------------------------|
| `typed-decisions` | `{state, questions, gold, agreement, synthetic}`  | native; Strands Decider-style models |
| `unsloth`         | `{instruction, input, output}` (Alpaca)           | Unsloth / TRL / Axolotl SFT          |
| `conversational`  | `{conversations: [user, assistant]}` (ShareGPT)   | Unsloth's recommended default; TRL   |
| `raw`             | `{state, gold}`                                   | minimal / custom harnesses           |

Every export also writes a `<out>.spec.json` sidecar with the schema.

**Training handoff (`--emit-helper`).** For `unsloth` and `conversational`, add
`--emit-helper` to also generate a ready-to-run `<out>.train.py`. It contains the
`formatting_prompts_func` + `dataset.map(...)` + `SFTTrainer` boilerplate, so the
only thing you edit is the base model id. The chat-template step still runs in
your training environment (it needs your model's tokenizer); the helper just
writes that glue for you.

```bash
s1-ready prepare --use-case "..." --corpus tickets.txt \
  --format conversational --emit-helper --out train.jsonl
# writes: train.jsonl, train.jsonl.spec.json, train.jsonl.train.py
```

## Teachers (model-agnostic)

The teacher drives both schema inference and labeling. The prompts and parsing
are plain, vendor-neutral text — **any instruction-following LLM works**, not
just OpenAI. Pick one with `--teacher` / `teacher=`:

| `teacher`                              | Backend                                                        |
|----------------------------------------|----------------------------------------------------------------|
| `heuristic` (default)                  | zero-dependency, offline, no key                               |
| `openai:gpt-4o-mini`                   | OpenAI (reads `OPENAI_API_KEY`)                                |
| `openai:<model>@http://localhost:11434/v1` | any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Together, Groq, Azure) — or set `OPENAI_BASE_URL` |
| `anthropic:claude-haiku-5-5`           | Anthropic direct (reads `ANTHROPIC_API_KEY`) — no AWS needed; use whatever id `models.list()` returns for your account |
| `bedrock:anthropic.claude-3-haiku-...` | AWS Bedrock (reads AWS creds)                                  |
| a Python callable `fn(prompt)->str`    | bring your own: local HF pipeline, internal gateway, anything  |

```python
# Local model via Ollama (OpenAI-compatible), no cloud, no key:
spec = SchemaInferencer.from_prompt(use_case, teacher="openai:llama3@http://localhost:11434/v1")

# Fully custom backend — the real escape hatch from any vendor:
def my_model(prompt: str) -> str: ...     # call whatever you like
spec = SchemaInferencer.from_prompt(use_case, teacher=my_model)
```

If a teacher is unavailable or returns unusable output, schema inference falls
back to the heuristic rather than failing. The 96/100 LLM benchmark above used
`openai:gpt-4o-mini`, but nothing in the prompts is tuned to it.
