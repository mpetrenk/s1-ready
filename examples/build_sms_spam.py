"""Build a System 1-ready dataset from the UCI SMS Spam Collection.

Source : UCI Machine Learning Repository, "SMS Spam Collection" (CC BY 4.0)
         https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Decision: is_spam (bool) -- is this SMS message spam?

This uses the dataset's REAL human gold labels (not the LLM teacher), so the
export is free, deterministic, and reproducible. It constructs s1-ready
`Spec` + `Row` objects directly and exports via the library's conversational
(ShareGPT) adapter plus the Unsloth train helper.

Usage:
    python examples/build_sms_spam.py /path/to/SMSSpamCollection

Produces (next to --out):
    sms_spam.jsonl             ShareGPT rows: {conversations: [...]}
    sms_spam.jsonl.spec.json   schema sidecar
    sms_spam.jsonl.train.py    ready-to-run Unsloth trainer
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from s1_ready.types import Question, QuestionType, Spec, Row
from s1_ready.gates import Dataset


def load_uci_tsv(path: str):
    """Yield (label, text) from the UCI TSV (label<TAB>text per line)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            label = label.strip().lower()
            text = text.strip()
            if label in ("ham", "spam") and text:
                yield label, text


def build_dataset(pairs, balance: bool, seed: int) -> Dataset:
    spec = Spec(
        use_case="Decide whether an SMS message is spam",
        questions=[
            Question(
                name="is_spam",
                type=QuestionType.BOOL,
                prompt="Is this SMS message spam?",
            )
        ],
    )

    rows = [
        Row(
            state=text,
            gold={"is_spam": (label == "spam")},
            agreement={"is_spam": 1.0},  # human gold label: full confidence
            synthetic=False,
        )
        for label, text in pairs
    ]

    if balance:
        rng = random.Random(seed)
        spam = [r for r in rows if r.gold["is_spam"]]
        ham = [r for r in rows if not r.gold["is_spam"]]
        k = min(len(spam), len(ham))
        rng.shuffle(spam)
        rng.shuffle(ham)
        rows = spam[:k] + ham[:k]
        rng.shuffle(rows)

    return Dataset(spec=spec, rows=rows)


def stratified_split(pairs, holdout: float, seed: int):
    """Split (label, text) pairs into (train, test), stratified by label.

    The split happens on the RAW pairs before any balancing, so train and test
    never share a message -- a clean held-out set for honest evaluation.
    """
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a spam-detection dataset for Colab/Unsloth.")
    p.add_argument("tsv", help="path to UCI SMSSpamCollection file")
    p.add_argument("--out", default="examples/sms_spam.jsonl")
    p.add_argument("--balance", action="store_true", help="down-sample ham to match spam count (train only)")
    p.add_argument("--limit", type=int, default=0, help="cap total train rows (0 = all)")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="fraction held out as a disjoint test set, written to <out>.test.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    pairs = list(load_uci_tsv(args.tsv))
    if not pairs:
        print(f"no (label, text) pairs read from {args.tsv}", file=sys.stderr)
        return 2

    # Deduplicate by message text BEFORE splitting. The UCI corpus contains
    # repeated messages; if the same text landed in both train and test it would
    # contaminate the held-out set and inflate eval numbers. Keep first occurrence.
    seen = set()
    deduped = []
    for label, text in pairs:
        if text not in seen:
            seen.add(text)
            deduped.append((label, text))
    if len(deduped) != len(pairs):
        print(f"deduped {len(pairs) - len(deduped)} repeated message(s): {len(pairs)} -> {len(deduped)}")
    pairs = deduped

    test_pairs = []
    if args.holdout > 0:
        pairs, test_pairs = stratified_split(pairs, args.holdout, args.seed)

    ds = build_dataset(pairs, balance=args.balance, seed=args.seed)
    if args.limit > 0:
        ds.rows = ds.rows[: args.limit]

    # Train export: ShareGPT (conversational) + Unsloth train helper.
    out = ds.to_conversational(args.out, emit_helper=True)
    spam = sum(1 for r in ds.rows if r.gold["is_spam"])
    print(f"TRAIN: wrote {len(ds)} rows -> {out}")
    print(f"  spam={spam}  ham={len(ds) - spam}")
    print(f"  schema sidecar: {out}.spec.json")
    print(f"  train helper:   {out}.train.py")

    # Test export: disjoint, NOT balanced (reflects real class ratio), gold kept.
    if test_pairs:
        test_ds = build_dataset(test_pairs, balance=False, seed=args.seed)
        test_out = f"{os.path.splitext(args.out)[0]}.test.jsonl"
        test_ds.to_conversational(test_out, emit_helper=False)
        tspam = sum(1 for r in test_ds.rows if r.gold["is_spam"])
        print(f"TEST:  wrote {len(test_ds)} rows -> {test_out}")
        print(f"  spam={tspam}  ham={len(test_ds) - tspam}  (disjoint from train, unbalanced)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
