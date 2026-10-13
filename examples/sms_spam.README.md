# SMS Spam demo dataset (`sms_spam.jsonl`)

A ready-to-train, System 1-ready dataset for a single `bool` decision:
**is this SMS message spam?**

## Provenance & license

- **Source:** UCI Machine Learning Repository — *SMS Spam Collection* (Almeida, T.A. & Gómez Hidalgo, J.M.).
  <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0) — free to use with attribution.
- **Labels:** the dataset's **real human gold labels** (ham/spam), not LLM-generated. The
  teacher/consensus path was intentionally not used here, so this export is free,
  deterministic, and faithful to the source labels.

## What's here

| File                        | What it is                                             |
|-----------------------------|--------------------------------------------------------|
| `sms_spam.jsonl`            | 1,494 rows, ShareGPT format, **balanced** 747 spam / 747 ham |
| `sms_spam.jsonl.spec.json`  | the decision schema (`is_spam: bool`)                  |
| `sms_spam.jsonl.train.py`   | ready-to-run Unsloth LoRA trainer                      |

Row shape:

```json
{"conversations": [
  {"role": "user", "content": "Is this SMS message spam? (true or false)\n\n<message>"},
  {"role": "assistant", "content": "is_spam: true"}
]}
```

## Reproduce

```bash
curl -sL -o sms.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip sms.zip
PYTHONPATH=. python3 examples/build_sms_spam.py SMSSpamCollection \
  --balance --seed 7 --out examples/sms_spam.jsonl
```

## Train on free Colab

1. colab.research.google.com -> New notebook -> Runtime -> T4 GPU.
2. Upload `sms_spam.jsonl`.
3. `!pip install -q unsloth trl datasets`
4. Paste the body of `sms_spam.jsonl.train.py` (set `MODEL_NAME`, point `DATA_FILE` at the upload).
5. Run. 1 epoch on a T4 is a few minutes for ~1.5k short rows.

> Balanced down-sampling drops most ham to match the 747 spam, so this is a
> demo-sized set. For a stronger model, keep more ham (adjust `build_sms_spam.py`)
> and tune the class balance for your real traffic.
