"""Auto-labeling driver: the experiment that tests s1-ready's real promise.

Unlike `examples/build_sms_spam.py` (which trains on the UCI corpus's HUMAN
labels), this script HIDES the human labels and has `s1-ready` GENERATE the
training labels from the raw message text with an LLM teacher. The hidden human
labels are then used only as an answer key to measure how good the generated
labels are.

Protocol
--------
1. Load the UCI SMS Spam Collection, dedup by text, stratified 80/20 split
   (identical seed/holdout to build_sms_spam.py, so the test set matches).
2. Set the TRAIN labels aside (the "hidden gold"). The model never sees them.
3. Feed the UNLABELED train text through `s1-ready`'s Labeler + teacher(s).
4. Score the generated labels against the hidden gold:
   precision / recall / F1 / accuracy / disagreement rate, and the same again
   AFTER removing the review-queue rows (to show how much human review buys).
5. Export the s1-LABELED train set (ShareGPT + Unsloth helper) for fine-tuning,
   and a human-LABELED train set from the SAME rows for the reference arm.
6. Export the untouched human-labeled TEST set (shared by every arm).

What this proves (once fine-tuned + evaluated via eval_four_way.py): whether a
dataset s1-ready labeled automatically trains a model that rivals one trained on
human labels — the gap between those two arms is the price of automation.

Nothing here needs a GPU. It needs a teacher: an LLM teacher (API key) to test
the real claim, or `--teacher heuristic` (no key) to exercise the plumbing.

Usage
-----
    # Real experiment (needs OPENAI_API_KEY / ANTHROPIC_API_KEY):
    OPENAI_API_KEY=... python label_sms.py SMSSpamCollection \
        --teacher openai:gpt-4o-mini --out-dir out/

    # Two-teacher consensus:
    python label_sms.py SMSSpamCollection \
        --teacher openai:gpt-4o-mini anthropic:claude-haiku-5-5 --out-dir out/

    # Plumbing smoke test, no key (heuristic labels will score poorly — expected):
    python label_sms.py SMSSpamCollection --teacher heuristic --out-dir out/ --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

# Import s1-ready from the installed package (pip install -e .) — this is the
# library doing the actual labeling, which is the whole point of the study.
from s1_ready.forge import DataForge
from s1_ready.gates import Dataset
from s1_ready.labeler import Labeler
from s1_ready.types import Question, QuestionType, Row, Spec


SPEC = Spec(
    use_case="Decide whether an SMS message is spam",
    questions=[
        Question(
            name="is_spam",
            type=QuestionType.BOOL,
            prompt="Is this SMS message spam (an unsolicited promotional or "
            "fraudulent message), as opposed to a normal personal message?",
        )
    ],
)


def load_uci_tsv(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            label, text = label.strip().lower(), text.strip()
            if label in ("ham", "spam") and text:
                pairs.append((label, text))
    return pairs


def dedup(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for label, text in pairs:
        if text not in seen:
            seen.add(text)
            out.append((label, text))
    return out


def stratified_split(pairs, holdout: float, seed: int):
    rng = random.Random(seed)
    spam = [p for p in pairs if p[0] == "spam"]
    ham = [p for p in pairs if p[0] == "ham"]
    rng.shuffle(spam)
    rng.shuffle(ham)

    def cut(items):
        n_test = int(len(items) * holdout)
        return items[n_test:], items[:n_test]  # (train, test)

    spam_tr, spam_te = cut(spam)
    ham_tr, ham_te = cut(ham)
    train = spam_tr + ham_tr
    test = spam_te + ham_te
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def score_labels(
    generated: List[bool], hidden_gold: List[bool]
) -> Dict[str, float]:
    """Label-quality metrics: generated train labels vs. hidden human gold.

    Positive class = spam (True). Precision/recall are of the SPAM label.
    """
    assert len(generated) == len(hidden_gold)
    tp = fp = tn = fn = 0
    for g, y in zip(generated, hidden_gold):
        if g and y:
            tp += 1
        elif g and not y:
            fp += 1
        elif not g and not y:
            tn += 1
        else:
            fn += 1
    n = len(generated)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    disagree = (fp + fn) / n if n else 0.0
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "precision_spam": round(prec, 4),
        "recall_spam": round(rec, 4),
        "f1_spam": round(f1, 4),
        "disagreement_rate": round(disagree, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tsv", help="path to UCI SMSSpamCollection file")
    p.add_argument(
        "--teacher", nargs="+", default=["heuristic"],
        help="one or more teacher ids (e.g. openai:gpt-4o-mini anthropic:claude-haiku-5-5)",
    )
    p.add_argument("--out-dir", default="out")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--limit", type=int, default=0,
        help="cap the number of TRAIN rows actually sent to the teacher "
        "(0 = all). Use a small cap first to bound API cost.",
    )
    p.add_argument(
        "--review-threshold", type=float, default=0.7,
        help="rows with min agreement below this go to the review queue; "
        "label quality is reported with and without them.",
    )
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = dedup(load_uci_tsv(args.tsv))
    if not pairs:
        print(f"no (label, text) pairs read from {args.tsv}", file=sys.stderr)
        return 2
    train_pairs, test_pairs = stratified_split(pairs, args.holdout, args.seed)
    if args.limit > 0:
        train_pairs = train_pairs[: args.limit]

    # Hidden gold: set the human train labels aside. The teacher never sees them.
    train_texts = [t for _, t in train_pairs]
    hidden_gold = [lab == "spam" for lab, _ in train_pairs]

    print(f"train rows to label: {len(train_texts)}  "
          f"(hidden gold: {sum(hidden_gold)} spam / {len(hidden_gold) - sum(hidden_gold)} ham)")
    print(f"teacher(s): {args.teacher}")

    # --- THE STEP UNDER TEST: s1-ready generates the labels ----------------
    forge = DataForge(SPEC).ingest(train_texts)
    labeler = Labeler(teacher=args.teacher)
    labeled = labeler.label(forge)  # LabeledRows

    generated = [bool(r.gold["is_spam"]) for r in labeled.rows]

    # --- Measure label quality vs. hidden gold -----------------------------
    overall = score_labels(generated, hidden_gold)

    # After human review: drop the low-agreement rows (what review would flag).
    review_idx = {
        i for i, r in enumerate(labeled.rows)
        if r.min_agreement < args.review_threshold
    }
    kept = [(generated[i], hidden_gold[i]) for i in range(len(generated)) if i not in review_idx]
    reviewed = score_labels([g for g, _ in kept], [y for _, y in kept]) if kept else None

    report = {
        "teacher": args.teacher,
        "n_train": len(train_texts),
        "review_threshold": args.review_threshold,
        "n_sent_to_review_queue": len(review_idx),
        "label_quality_all_rows": overall,
        "label_quality_after_review": reviewed,
        "agreement_summary": labeled.agreement_summary(),
        "note": (
            "label_quality_* compares s1-ready's GENERATED train labels against "
            "the hidden UCI human labels. 'after_review' excludes rows whose "
            "teacher agreement fell below review_threshold (the rows a human "
            "would check). Positive class = spam."
        ),
    }
    report_path = os.path.join(args.out_dir, "label_quality.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== teacher-label quality vs. hidden human gold ===")
    print(f"  all rows      : {overall}")
    if reviewed:
        print(f"  after review  : {reviewed}  "
              f"(dropped {len(review_idx)} low-agreement rows)")
    print(f"  report written: {report_path}")

    # --- Export the two training arms + the shared test set ----------------
    # Arm A: s1-LABELED train (what the library produced). This is the dataset
    # whose fine-tune we actually care about.
    auto_ds = Dataset(spec=SPEC, rows=list(labeled.rows))
    auto_out = os.path.join(args.out_dir, "train_auto.jsonl")
    auto_ds.to_conversational(auto_out, emit_helper=True)

    # Arm B: HUMAN-LABELED train from the SAME rows (reference arm). Same texts,
    # same order, gold swapped in — isolates label quality as the only variable.
    human_ds = Dataset(
        spec=SPEC,
        rows=[
            Row(state=t, gold={"is_spam": y}, agreement={"is_spam": 1.0}, synthetic=False)
            for t, y in zip(train_texts, hidden_gold)
        ],
    )
    human_out = os.path.join(args.out_dir, "train_human.jsonl")
    human_ds.to_conversational(human_out, emit_helper=False)

    # Shared TEST set: untouched human labels, real class ratio, disjoint.
    test_ds = Dataset(
        spec=SPEC,
        rows=[
            Row(state=t, gold={"is_spam": lab == "spam"}, agreement={"is_spam": 1.0})
            for lab, t in test_pairs
        ],
    )
    test_out = os.path.join(args.out_dir, "test_human.jsonl")
    test_ds.to_conversational(test_out, emit_helper=False)

    tspam = sum(1 for lab, _ in test_pairs if lab == "spam")
    print("\n=== datasets written ===")
    print(f"  Arm A (s1 auto-labeled train): {auto_out}  (+ .train.py, .spec.json)")
    print(f"  Arm B (human-labeled train)  : {human_out}")
    print(f"  Shared test (human gold)     : {test_out}  "
          f"({tspam} spam / {len(test_pairs) - tspam} ham, disjoint)")
    print("\nNext: fine-tune on each train arm, then run eval_four_way.py on the "
          "shared test set. See auto_label/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
