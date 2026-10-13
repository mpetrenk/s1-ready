"""Assemble the Colab notebook for the SMS-spam case study.

Builds s1_spam_colab.ipynb from clean cell sources so the notebook stays
consistent with the library's export format and the eval/compare logic.
Run:  python examples/case_study/_build_notebook.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "s1_spam_colab.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""# s1-ready → Unsloth: SMS spam case study

End-to-end on a **free Colab T4 GPU**: upload a dataset built by `s1-ready`,
fine-tune Llama-3.1 with a LoRA, evaluate on a held-out set, and compare against
the untrained base model and a classic TF-IDF + Naive Bayes baseline.

**Before you start:** Runtime → Change runtime type → **T4 GPU**.

**Build the data on your machine first** (needs `s1-ready` + the UCI corpus):

```bash
python examples/build_sms_spam.py SMSSpamCollection \\
  --balance --holdout 0.2 --seed 7 --out examples/sms_spam.jsonl
```

That produces `sms_spam.jsonl` (train) and `sms_spam.test.jsonl` (held-out).
Upload both in the next cell."""),

    md("## 1. Upload the dataset\nPick `sms_spam.jsonl`, then `sms_spam.test.jsonl`."),
    code("""
from google.colab import files
files.upload()  # upload sms_spam.jsonl and sms_spam.test.jsonl
"""),

    md("## 2. Install Unsloth + TRL\n_If this errors, grab the current install line from Unsloth's official Llama-3.1 Colab and keep the rest unchanged._"),
    code("!pip install -q unsloth trl datasets scikit-learn"),

    md("## 3. Load the model, format with the chat template, and train\nLoRA fine-tune, 1 epoch — a few minutes on a T4. Watch the loss drop."),
    code("""
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = standardize_sharegpt(load_dataset("json", data_files="sms_spam.jsonl", split="train"))
def fmt(ex):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                     for c in ex["conversations"]]}
ds = ds.map(fmt, batched=True)
print("=== sample rendered row ==="); print(ds[0]["text"])

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, num_train_epochs=1, learning_rate=2e-4,
        logging_steps=10, output_dir="outputs"))
trainer.train()
"""),

    md("## 4. Evaluate on the held-out set\nAccuracy / precision / recall / F1 + confusion matrix (spam = positive)."),
    code("""
import json, re, warnings, transformers
transformers.logging.set_verbosity_error(); warnings.filterwarnings("ignore")
FastLanguageModel.for_inference(model)

TEST_FILE = "sms_spam.test.jsonl"
rows = [json.loads(l) for l in open(TEST_FILE)]

def gold_of(r): return "true" in r["conversations"][1]["content"].lower()

def predict(user_content):
    msgs = [{"role": "user", "content": user_content}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    m = re.search(r"is_spam:\\s*(true|false)", txt, re.I)
    return (m.group(1).lower() == "true") if m else False

tp = tn = fp = fn = 0
for i, r in enumerate(rows):
    g, p = gold_of(r), predict(r["conversations"][0]["content"])
    if   g and p: tp += 1
    elif g:       fn += 1
    elif p:       fp += 1
    else:         tn += 1
    if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(rows)}")

n = tp+tn+fp+fn
acc = (tp+tn)/n; prec = tp/(tp+fp) if tp+fp else 0
rec = tp/(tp+fn) if tp+fn else 0; f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
print(f"\\naccuracy {acc:.3f}  precision {prec:.3f}  recall {rec:.3f}  f1 {f1:.3f}")
print(f"confusion: TP={tp} FN={fn} FP={fp} TN={tn}")
"""),

    md("## 5. Compare: fine-tuned vs base LLM vs classic ML\nAll three on the **same** held-out test set."),
    code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

golds = [gold_of(r) for r in rows]
def sms_of(r): return r["conversations"][0]["content"].split("\\n\\n", 1)[-1]
def score(name, preds):
    a = sum(p==g for p,g in zip(preds,golds))/len(golds)
    TP = sum(p and g for p,g in zip(preds,golds)); FP = sum(p and not g for p,g in zip(preds,golds))
    FN = sum((not p) and g for p,g in zip(preds,golds))
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    F = 2*P*R/(P+R) if P+R else 0
    return {"model": name, "acc": a, "prec": P, "rec": R, "f1": F}

results = [{"model":"Llama-3.1 FT (s1-ready)","acc":acc,"prec":prec,"rec":rec,"f1":f1}]

# base model, no fine-tune
base, base_tok = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
FastLanguageModel.for_inference(base)
def base_predict(uc):
    ids = base_tok.apply_chat_template([{"role":"user","content":uc}], add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = base.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=base_tok.eos_token_id)
    txt = base_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    m = re.search(r"(true|false|spam|ham)", txt, re.I)
    return bool(m) and m.group(1).lower() in ("true","spam")
print("scoring base model...")
results.append(score("Llama-3.1 base (no FT)", [base_predict(r["conversations"][0]["content"]) for r in rows]))

# classic ML
train_rows = [json.loads(l) for l in open("sms_spam.jsonl")]
clf = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2), MultinomialNB())
clf.fit([sms_of(r) for r in train_rows], [gold_of(r) for r in train_rows])
results.append(score("TF-IDF + NaiveBayes", list(clf.predict([sms_of(r) for r in rows]))))

print(f"\\n{'model':<26} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
print("-"*54)
for r in results:
    print(f"{r['model']:<26} {r['acc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")
"""),

    md("""## What to take away

- The **same base model**, prompted the same way, goes from ~0.30 F1 untrained to
  ~0.96 fine-tuned. The dataset `s1-ready` produced is the entire difference.
- On this easy task, **TF-IDF + Naive Bayes is close** (and ~1000× cheaper). The
  LLM's edge shows most in precision. The s1-ready/LLM route earns its cost on
  *harder* decisions (many options, nuance, no hand-engineered features).
- Full writeup: `CASE_STUDY_sms_spam.md`."""),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(NB, fh, indent=1, ensure_ascii=False)
print(f"wrote {OUT}")
