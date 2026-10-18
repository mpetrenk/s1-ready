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

## Results — label quality (measured)

Ran on the full UCI train split (4,129 messages after dedup, 514 spam / 3,615
ham), two teachers: `openai:gpt-4o-mini` + `anthropic:claude-haiku-5-5`. The
hidden human labels are the answer key; positive class = spam.

| | rows | precision | recall | F1 | disagreement |
|---|---:|---:|---:|---:|---:|
| **Raw consensus labels** | 4,129 | 0.450 | 0.977 | 0.616 | 15.1% |
| **After review** (drop disagreements) | 3,556 | 0.846 | 0.984 | **0.910** | 2.8% |

**What this shows:** a single-pass LLM label is mediocre — raw auto-labels hit
only **F1 0.62**, over-flagging ham as spam (613 false positives). But the
**agreement signal localizes those errors**: dropping the 573 rows (14%) where
the two teachers disagreed removes 523 of the 613 false positives, taking
precision **0.45 → 0.85** and **F1 0.62 → 0.91** — with recall essentially
unchanged (0.98). This is the review queue doing its job: it does not fix labels,
it tells you *which* 14% to check (or discard) so the remaining 86% is near-clean
training data.

**Honest caveats:**
- *A filter, not a correction.* The 0.91 is on the 3,556 kept rows; the 573
  flagged rows are set aside, not repaired. A human reviewing them recovers more.
- *~90 false positives remain* in the kept set (precision 0.85). Good, not perfect
  — a model trained on it inherits that.
- *Both teachers share an over-flagging bias*, so raw-consensus precision barely
  beat a single teacher (0.45 vs 0.41 at n=300). Consensus helped only via the
  **disagreement signal**, not by averaging. On this task that's what mattered.
- *SMS spam is lexically easy*, which inflates recall. The gate's value should be
  larger on harder, more ambiguous tasks — untested here.

Reproduce: `label_sms.py SMSSpamCollection --teacher openai:gpt-4o-mini anthropic:claude-haiku-5-5 --out-dir out_full`
(the committed `out_full/label_quality.json` holds these numbers).

## Status — fine-tune arm pending

Label quality is measured (above). The **four-way fine-tune comparison**
(base / FT-auto / FT-human / classical) still needs a GPU run via
[`auto_label_colab.ipynb`](auto_label_colab.ipynb) to answer the last question:
does training on these agreement-gated auto-labels rival training on human
labels? Those numbers get committed here when the run lands.
