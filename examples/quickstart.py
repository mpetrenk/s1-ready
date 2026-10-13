"""End-to-end library usage of s1-ready on the sample ticket corpus.

Run from the repo root:

    python examples/quickstart.py

Produces examples/train.jsonl (+ a .spec.json sidecar).
"""

import os

from s1_ready import SchemaInferencer, DataForge, Labeler

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "tickets.txt")
OUT = os.path.join(HERE, "train.jsonl")


def main() -> None:
    with open(CORPUS) as fh:
        spec = SchemaInferencer.from_prompt(
            "Route support tickets to billing, account, sales, or technical, and flag refunds",
            sample_states=fh,
        )
    print("Inferred schema:")
    for q in spec.questions:
        extra = f" options={q.options}" if q.options else ""
        print(f"  - {q.name} [{q.type.value}]{extra}")

    forge = DataForge(spec, seed=7)
    forge.ingest(CORPUS)
    forge.synthesize(n=12, balance=True)
    print(f"\nPool: {len(forge)} states ({sum(s.synthetic for s in forge.states)} synthetic)")

    # Two salted heuristic teachers so consensus/agreement is non-trivial offline.
    rows = Labeler(teacher=["heuristic", "heuristic-b"], vote="majority").label(forge)
    print(f"Labeled {len(rows)} rows; mean agreement {rows.agreement_summary()}")
    print(f"Review queue (<0.7): {len(rows.review_queue())} rows")

    ds = (
        rows._as_dataset()
        .balance()
        .confidence_filter(min=0.0)  # keep all for the demo
    )
    print(f"Gold distribution: {ds.distribution()}")

    out = ds.to_jsonl(OUT, fmt="typed-decisions")
    print(f"\nWrote {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
