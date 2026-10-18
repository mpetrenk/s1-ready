"""Four-way evaluation on the shared, untouched human-labeled test set.

Separates two questions the single-arm study conflated:
  - Does fine-tuning help?            (base vs. any fine-tune)
  - Are the AUTO labels good enough?  (auto-label FT vs. human-label FT)

Arms, all scored on the SAME test_human.jsonl (real class ratio, disjoint):
  1. base            - the un-fine-tuned base model, prompted zero-shot
  2. ft_auto         - fine-tuned on s1-ready's GENERATED labels (train_auto)
  3. ft_human        - fine-tuned on human labels, same texts (train_human)
  4. classical       - TF-IDF + Naive Bayes trained on train_human

The headline number is the gap between (2) and (3): the price of automation.
If ft_auto ≈ ft_human, s1-ready's labeling is "good enough to train on" — which
is the library's whole thesis.

This file is import-safe with no ML deps installed: the metric helpers and the
classical baseline (scikit-learn) stand alone, and the LLM-prediction functions
are only called when you pass live models in (from the Colab notebook, which has
GPU + Unsloth). Run `python eval_four_way.py --selftest` to check the scoring
math with no data or models.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Callable, Dict, List, Optional


# --- metrics ---------------------------------------------------------------

def binary_metrics(preds: List[bool], gold: List[bool]) -> Dict[str, float]:
    """Spam = positive class. Returns accuracy/precision/recall/f1 + confusion."""
    assert len(preds) == len(gold), "preds and gold must align"
    tp = fp = tn = fn = 0
    for p, y in zip(preds, gold):
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif not p and not y:
            tn += 1
        else:
            fn += 1
    n = len(preds)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# --- data loading ----------------------------------------------------------

def load_conversational(path: str):
    """Read a ShareGPT jsonl written by s1-ready; return (texts, gold_bools).

    The user turn carries the state; the assistant turn carries the gold answer
    as JSON like {"is_spam": true}. We parse the boolean back out.
    """
    texts: List[str] = []
    gold: List[bool] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            convs = rec.get("conversations", [])
            user = next((c["content"] for c in convs if c.get("role") == "user"), "")
            asst = next((c["content"] for c in convs if c.get("role") == "assistant"), "")
            texts.append(user)
            gold.append(_parse_is_spam(asst))
    return texts, gold


def _parse_is_spam(answer: str) -> bool:
    """Pull the is_spam boolean out of an assistant answer string.

    s1-ready's conversational export writes the answer as ``is_spam: true`` /
    ``is_spam: false`` (NOT JSON). Strip the ``is_spam`` field name first so the
    literal "spam" inside it can't be mistaken for the answer, then read the
    boolean. A JSON ``{"is_spam": true}`` form is also accepted as a fallback.
    """
    a = answer.strip().lower().replace("is_spam", "").replace("_", " ")
    m = re.search(r"\b(true|false|yes|no)\b", a)
    if m:
        return m.group(1) in ("true", "yes")
    try:
        obj = json.loads(re.search(r"\{.*\}", answer, re.DOTALL).group())
        return bool(obj.get("is_spam"))
    except Exception:
        return False


# --- arms ------------------------------------------------------------------

def eval_classical(train_path: str, test_path: str) -> Dict[str, float]:
    """TF-IDF + Multinomial Naive Bayes. Trains in seconds on CPU."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    tr_x, tr_y = load_conversational(train_path)
    te_x, te_y = load_conversational(test_path)
    clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
    clf.fit(tr_x, [1 if y else 0 for y in tr_y])
    preds = [bool(p) for p in clf.predict(te_x)]
    return binary_metrics(preds, te_y)


def eval_llm(
    predict_fn: Callable[[List[str]], List[bool]], test_path: str
) -> Dict[str, float]:
    """Score any model exposed as predict_fn(texts) -> list[bool].

    The Colab notebook builds `predict_fn` for the base model and for each
    fine-tuned adapter (Unsloth) and passes it here. Keeping prediction out of
    this file means the scoring logic has zero ML dependencies.
    """
    te_x, te_y = load_conversational(test_path)
    preds = predict_fn(te_x)
    return binary_metrics(preds, te_y)


def summarize(results: Dict[str, Dict[str, float]]) -> str:
    """Render the four-way table + the two interpretive gaps."""
    rows = ["| arm | accuracy | precision | recall | F1 |",
            "|---|---:|---:|---:|---:|"]
    labels = {
        "base": "Base (no fine-tune)",
        "ft_auto": "Fine-tuned on AUTO labels (raw, all rows)",
        "ft_auto_gated": "Fine-tuned on AUTO labels (agreement-gated)",
        "ft_human": "Fine-tuned on HUMAN labels",
        "classical": "TF-IDF + Naive Bayes",
    }
    for key in ("ft_auto", "ft_auto_gated", "ft_human", "base", "classical"):
        if key in results:
            m = results[key]
            rows.append(f"| {labels[key]} | {m['accuracy']} | {m['precision']} "
                        f"| {m['recall']} | {m['f1']} |")
    out = "\n".join(rows)

    # Best available auto arm (gated preferred) vs. human reference.
    best_auto = "ft_auto_gated" if "ft_auto_gated" in results else "ft_auto"
    if best_auto in results and "ft_human" in results:
        gap = round(results["ft_human"]["f1"] - results[best_auto]["f1"], 4)
        which = "gated" if best_auto == "ft_auto_gated" else "raw"
        out += (f"\n\n**Price of automation (F1):** human-label FT − {which} "
                f"auto-label FT = {gap:+.4f}. Near zero means s1-ready's automatic "
                f"labels train a model as good as human labels.")
    # Downstream payoff of the review queue.
    if "ft_auto_gated" in results and "ft_auto" in results:
        lift = round(results["ft_auto_gated"]["f1"] - results["ft_auto"]["f1"], 4)
        out += (f"\n\n**Review-queue payoff (F1):** gated − raw auto = {lift:+.4f}. "
                f"Positive means dropping the disagreement rows before training "
                f"produced a better model.")
    return out


def _selftest() -> int:
    # 8 preds vs gold with a known confusion matrix.
    preds = [True, True, False, False, True, False, True, False]
    gold  = [True, False, False, True, True, False, False, False]
    m = binary_metrics(preds, gold)
    assert m["tp"] == 2 and m["fp"] == 2 and m["tn"] == 3 and m["fn"] == 1, m
    assert m["accuracy"] == round(5 / 8, 4), m
    assert m["precision"] == round(2 / 4, 4), m
    assert m["recall"] == round(2 / 3, 4), m
    # The export format: "is_spam: true/false". The field name contains the
    # literal "spam" -- these cases guard against reading it as the answer.
    assert _parse_is_spam("is_spam: true") is True
    assert _parse_is_spam("is_spam: false") is False
    assert _parse_is_spam("IS_SPAM: FALSE") is False
    assert _parse_is_spam("is_spam:true") is True
    # JSON fallback still works.
    assert _parse_is_spam('{"is_spam": true}') is True
    assert _parse_is_spam('{"is_spam": false}') is False
    print("eval_four_way selftest: OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true",
                   help="verify the scoring math with no data/models and exit")
    p.add_argument("--train-human", help="train_human.jsonl (for classical arm)")
    p.add_argument("--test", help="test_human.jsonl (shared test set)")
    p.add_argument("--out", default="four_way_results.json")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    results: Dict[str, Dict[str, float]] = {}
    if args.train_human and args.test:
        results["classical"] = eval_classical(args.train_human, args.test)
        print("classical:", results["classical"])

    # The LLM arms (base / ft_auto / ft_human) are evaluated from the notebook,
    # which has the models loaded; it calls eval_llm(...) and merges into this
    # results dict, then calls summarize(). See auto_label/README.md.
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    if results:
        print("\n" + summarize(results))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
