"""Benchmark the heuristic schema inferencer against gold expectations.

Scores three things per case:
  - count:   does the number of inferred questions match?
  - types:   does the multiset of inferred question types match?
  - options: for each gold-specified question index, does the inferred option
             set match exactly? (only checked where gold provides options)

A case PASSES only if all three hold. Prints aggregate accuracy, a per-type
breakdown, and every failure with a category so gaps are actionable.

Run:  python examples/schema_benchmark/run_benchmark.py
Exit code 0 if accuracy >= THRESHOLD, else 1 (usable as a CI gate).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from s1_ready import SchemaInferencer  # noqa: E402

THRESHOLD = 0.90
HERE = os.path.dirname(os.path.abspath(__file__))


def norm_opt(s: str) -> str:
    """Normalize an option for fair comparison: lowercase, underscores/hyphens
    to spaces, collapsed whitespace. So 'feature_request' == 'feature request'.
    (We do NOT drop words, so a real truncation like 'low' vs 'low priority'
    still counts as a miss.)
    """
    s = s.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def types_match(got_types, exp_types, binary: bool) -> bool:
    """Type multiset match, with an allowance for inherently-binary decisions.

    When `binary` is set, a `bool` answer and a single 2-option `choice` are
    treated as equivalent -- both are valid models of a yes/no distinction.
    """
    if Counter(got_types) == Counter(exp_types):
        return True
    if binary and len(got_types) == 1 and len(exp_types) == 1:
        return {got_types[0], exp_types[0]} <= {"bool", "choice"}
    return False


TEACHER = os.environ.get("S1_BENCH_TEACHER")  # e.g. "openai:gpt-4o-mini"; None = heuristic


def check(case) -> dict:
    spec = SchemaInferencer.from_prompt(case["use_case"], teacher=TEACHER)
    got_types = [q.type.value for q in spec.questions]
    exp_types = case["types"]
    binary = bool(case.get("binary"))

    reasons = []
    if len(got_types) != len(exp_types):
        reasons.append(f"count: expected {len(exp_types)} got {len(got_types)} ({got_types})")
    if not types_match(got_types, exp_types, binary):
        reasons.append(f"types: expected {sorted(exp_types)} got {sorted(got_types)}")

    # options: only when the inferred question count matches (indices align).
    # A binary case answered as bool legitimately has no options -> skip.
    if "options" in case and len(got_types) == len(exp_types):
        for idx_s, exp_opts in case["options"].items():
            i = int(idx_s)
            got_opts = {norm_opt(o) for o in spec.questions[i].options}
            want = {norm_opt(o) for o in exp_opts}
            if binary and not got_opts:
                continue
            if got_opts != want:
                reasons.append(f"options[{i}]: expected {sorted(want)} got {sorted(got_opts)}")

    return {"use_case": case["use_case"], "ok": not reasons, "reasons": reasons,
            "exp_types": exp_types}


def main() -> int:
    data = json.load(open(os.path.join(HERE, "cases.json")))
    cases = data["cases"]
    results = [check(c) for c in cases]

    passed = sum(r["ok"] for r in results)
    total = len(results)
    acc = passed / total

    # per-type breakdown (by the dominant expected type of each case)
    by_type_total = Counter()
    by_type_pass = Counter()
    for r in results:
        key = "+".join(sorted(set(r["exp_types"])))
        by_type_total[key] += 1
        by_type_pass[key] += int(r["ok"])

    # failure categories
    cat = Counter()
    for r in results:
        for reason in r["reasons"]:
            cat[reason.split(":")[0]] += 1

    mode = f"LLM teacher={TEACHER}" if TEACHER else "heuristic (default)"
    print(f"=== Schema inferencer benchmark [{mode}]: {passed}/{total} = {acc:.1%} ===\n")
    print("By expected shape:")
    for key in sorted(by_type_total):
        print(f"  {key:<22} {by_type_pass[key]:>2}/{by_type_total[key]:<2}")
    if cat:
        print("\nFailure categories:")
        for k, v in cat.most_common():
            print(f"  {k:<10} {v}")
    fails = [r for r in results if not r["ok"]]
    if fails:
        print(f"\nFailures ({len(fails)}):")
        for r in fails:
            print(f"  - {r['use_case']}")
            for reason in r["reasons"]:
                print(f"      {reason}")

    print(f"\nthreshold {THRESHOLD:.0%} -> {'PASS' if acc >= THRESHOLD else 'FAIL'}")
    return 0 if acc >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
