# Auto-labeling experiment — testing the real promise

The main [SMS case study](../CASE_STUDY_sms_spam.md) trains on the UCI corpus's
**human** labels. It proves `s1-ready`'s export + fine-tuning path works, but it
does **not** test the library's headline claim: that `s1-ready` can *label an
unlabeled corpus* well enough to train a decision model.

This experiment tests exactly that.

## The protocol

```
UCI SMS corpus
   │  dedup + stratified 80/20 split (seed 7)
   ▼
train split ──[ HIDE the human labels ]──► kept only as an answer key
   │
   │  raw text (no labels) ──► s1-ready Labeler + LLM teacher(s)
   ▼
GENERATED labels  ──compare──►  hidden human labels   = label-quality report
   │
   ├──► train_auto.jsonl   (s1-ready's generated labels)   ─┐
   └──► train_human.jsonl  (human labels, same texts)       │ fine-tune each
                                                             ▼
test split (human labels, untouched) ◄── evaluate all 4 ──► four-way table
```

Four models, one shared untouched test set:

| arm | train labels | question it answers |
|---|---|---|
| **Base** | none (zero-shot) | baseline |
| **FT-auto** | **s1-ready generated** | *does the library's output train a usable model?* |
| **FT-human** | human gold | reference ceiling |
| **Classical** | human gold (TF-IDF + NB) | is the LLM route worth it at all? |

**The headline number is `F1(FT-human) − F1(FT-auto)` — the price of automation.**
Near zero means s1-ready's automatic labels are as good as human labels for
training, which is the whole thesis. We also report **teacher-label quality**
(generated labels vs. hidden gold: precision / recall / F1 / disagreement),
before and after dropping the review-queue rows.

## Why this is honest where the first study wasn't

The first study credited `s1-ready` with producing "the dataset" when the labels
were human. Here, `s1-ready` genuinely does the labeling, and the human labels
are demoted to a grading key. If the auto-label model is bad, this experiment
will show it — that's the point.

## Files

| file | what it does | needs |
|---|---|---|
| `label_sms.py` | hide labels → s1-ready labels → score vs hidden gold → export 3 datasets | an LLM key (or `--teacher heuristic` for a no-key plumbing test) |
| `eval_four_way.py` | metrics, classical baseline, LLM scoring hook, "price of automation" summary | scikit-learn (classical arm only) |
| `auto_label_colab.ipynb` | one-click: install → label → fine-tune both arms → evaluate | GPU + LLM key |

## Run it

### A) Labeling + label-quality, locally (no GPU)

```bash
pip install -e ".[llm]"          # from the repo root
cd examples/case_study/auto_label

# Download the corpus (UCI, CC BY 4.0):
curl -sL https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip -o sms.zip
unzip -o sms.zip

# The real experiment — needs a key. Start with --limit to bound API cost:
OPENAI_API_KEY=... python label_sms.py SMSSpamCollection \
    --teacher openai:gpt-4o-mini --out-dir out --limit 300

# Two-teacher consensus (also set ANTHROPIC_API_KEY):
python label_sms.py SMSSpamCollection \
    --teacher openai:gpt-4o-mini anthropic:claude-haiku-5-5 --out-dir out --limit 300

# No key? Exercise the plumbing with the heuristic teacher.
# (Labels will score poorly — the heuristic is not a real model. Expected.)
python label_sms.py SMSSpamCollection --teacher heuristic --out-dir out --limit 300
```

This writes `out/label_quality.json` plus `train_auto.jsonl`,
`train_human.jsonl`, `test_human.jsonl`. **Read `label_quality.json` first** — if
the generated labels don't match the hidden gold, no fine-tune will be good, and
you've learned that for a few cents instead of a GPU hour.

### B) Full end-to-end (GPU + key)

Open **[`auto_label_colab.ipynb`](auto_label_colab.ipynb)** in Colab (T4 GPU
runtime), add `OPENAI_API_KEY` via the Secrets panel, and run top to bottom. It
labels, fine-tunes both arms with Unsloth, evaluates all four, and prints the
price-of-automation table. Verify the scorer offline first:

```bash
python eval_four_way.py --selftest     # checks the metric math, no data/models
```

## Status

**Not yet run at scale.** The driver, scorer, and notebook are complete and the
scoring math is unit-tested, but the headline four-way numbers require a GPU +
API-key run. When that run lands, its `label_quality.json` and
`four_way_results.json` get committed here and the top-level README's "Does it
actually work?" table is updated to the auto-label numbers. Until then the repo
says plainly that only the human-label fine-tuning path is demonstrated.
