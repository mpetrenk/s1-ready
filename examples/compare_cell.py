# === Colab comparison cell ===================================================
# Run AFTER the eval cell (reuses `model`, `tokenizer`, and the TP/TN/FP/FN from
# your fine-tuned run). Compares three approaches on the SAME held-out test set:
#
#   1. Fine-tuned Llama-3.1 (s1-ready data)   <- already scored, reused here
#   2. Base Llama-3.1 instruct, NO fine-tune  <- same prompt, untrained
#   3. Classic ML: TF-IDF + Multinomial NB    <- trained on sms_spam.jsonl
#
# Prints a side-by-side table of accuracy / precision / recall / f1.

import json, re, warnings, transformers, torch
transformers.logging.set_verbosity_error(); warnings.filterwarnings("ignore")

TEST_FILE  = "sms_spam.test.jsonl"
TRAIN_FILE = "sms_spam.jsonl"
test_rows  = [json.loads(l) for l in open(TEST_FILE)]

def gold_of(r):  # spam -> True
    return "true" in r["conversations"][1]["content"].lower()
def user_of(r):
    return r["conversations"][0]["content"]
def sms_of(r):   # strip the "Is this... (true or false)\n\n" prefix -> raw message
    return user_of(r).split("\n\n", 1)[-1]

def score(name, preds, golds):
    tp = sum(p and g for p, g in zip(preds, golds))
    tn = sum((not p) and (not g) for p, g in zip(preds, golds))
    fp = sum(p and (not g) for p, g in zip(preds, golds))
    fn = sum((not p) and g for p, g in zip(preds, golds))
    n = tp + tn + fp + fn
    acc  = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"model": name, "acc": acc, "prec": prec, "rec": rec, "f1": f1}

golds = [gold_of(r) for r in test_rows]
results = []

# --- 1. Fine-tuned model: reuse the counts already computed in the eval cell ---
# (tp, tn, fp, fn are still in scope from eval_cell.py)
try:
    n = tp + tn + fp + fn
    results.append({
        "model": "Llama-3.1 FT (s1-ready)",
        "acc": (tp + tn) / n, "prec": tp / (tp + fp) if (tp+fp) else 0,
        "rec": tp / (tp + fn) if (tp+fn) else 0,
        "f1": (2*tp) / (2*tp + fp + fn) if (2*tp+fp+fn) else 0,
    })
except NameError:
    print("(fine-tuned counts not in scope -- run the eval cell first)")

# --- 2. Base Llama-3.1 (no fine-tune): same prompt, untrained ---------------
from unsloth import FastLanguageModel
base, base_tok = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.1-8b-instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(base)

def base_predict(user_content):
    msgs = [{"role": "user", "content": user_content}]
    ids = base_tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = base.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=base_tok.eos_token_id)
    txt = base_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    m = re.search(r"(true|false|spam|ham)", txt, re.I)
    if not m:
        return False
    tok = m.group(1).lower()
    return tok in ("true", "spam")

print("scoring base model (no fine-tune)...")
base_preds = [base_predict(user_of(r)) for r in test_rows]
results.append(score("Llama-3.1 base (no FT)", base_preds, golds))

# --- 3. Classic ML: TF-IDF + Multinomial Naive Bayes ------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

train_rows = [json.loads(l) for l in open(TRAIN_FILE)]
Xtr = [sms_of(r) for r in train_rows]
ytr = [gold_of(r) for r in train_rows]
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), MultinomialNB())
clf.fit(Xtr, ytr)
nb_preds = list(clf.predict([sms_of(r) for r in test_rows]))
results.append(score("TF-IDF + NaiveBayes", nb_preds, golds))

# --- Side-by-side table -----------------------------------------------------
print("\n=== Comparison on held-out test set (spam = positive) ===")
print(f"{'model':<26} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
print("-" * 54)
for r in results:
    print(f"{r['model']:<26} {r['acc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")
# =============================================================================
