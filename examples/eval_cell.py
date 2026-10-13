# === Colab eval cell =========================================================
# Run AFTER training (the model/tokenizer from the train cell are in memory).
# Upload sms_spam.test.jsonl first (held-out, disjoint from train).
#
# Scores the fine-tuned model on the held-out set: accuracy, precision, recall,
# F1, confusion matrix. Parses the model's "is_spam: true/false" line and
# compares to gold.

import json, re, torch, warnings
import transformers
from unsloth import FastLanguageModel
import transformers
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

# Silence the per-call "max_new_tokens vs max_length" warning so the loop
# output stays readable (it's benign: max_new_tokens correctly takes precedence).
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

FastLanguageModel.for_inference(model)   # `model`, `tokenizer` from the train cell
TEST_FILE = "sms_spam.test.jsonl"

rows = [json.loads(l) for l in open(TEST_FILE)]
# Optional: cap for speed on free Colab (full set ~1k rows is a few minutes).
# rows = rows[:300]

def gold_of(row):
    return "true" in row["conversations"][1]["content"].lower()

def predict(user_content):
    msgs = [{"role": "user", "content": user_content}]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    m = re.search(r"is_spam:\s*(true|false)", text, re.I)
    if m:
        return m.group(1).lower() == "true", text.strip()
    # Unparseable -> count as the negative class, flag it.
    return False, f"[UNPARSED] {text.strip()}"

tp = tn = fp = fn = unparsed = 0
for i, row in enumerate(rows):
    g = gold_of(row)
    p, raw = predict(row["conversations"][0]["content"])
    if raw.startswith("[UNPARSED]"):
        unparsed += 1
    if   g and p:     tp += 1
    elif g and not p: fn += 1
    elif not g and p: fp += 1
    else:             tn += 1
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{len(rows)}")

n = tp + tn + fp + fn
acc  = (tp + tn) / n if n else 0.0
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = tp / (tp + fn) if (tp + fn) else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

print("\n=== Held-out eval (spam = positive) ===")
print(f"n={n}  unparsed={unparsed}")
print(f"accuracy  {acc:.3f}")
print(f"precision {prec:.3f}")
print(f"recall    {rec:.3f}")
print(f"f1        {f1:.3f}")
print("confusion matrix:")
print(f"             pred_spam  pred_ham")
print(f"  gold_spam  {tp:>8}  {fn:>8}")
print(f"  gold_ham   {fp:>8}  {tn:>8}")
# =============================================================================
