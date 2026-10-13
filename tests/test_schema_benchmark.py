"""Regression gate: the schema inferencer must pass the 100-case benchmark.

The benchmark dataset (examples/schema_benchmark/cases.json) is the source of
truth for expected behavior across representative decision use cases. This test
runs it and fails if accuracy drops below the committed threshold, so a future
change to the heuristics can't silently regress schema quality.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from s1_ready import SchemaInferencer

BENCH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "schema_benchmark", "cases.json"
)
THRESHOLD = 0.95  # we currently pass 100/100; guard against drift


def _norm(s: str) -> str:
    # Match the scorer: lowercase, underscores/hyphens -> space, collapse ws.
    s = s.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def _types_match(got, exp, binary):
    if Counter(got) == Counter(exp):
        return True
    if binary and len(got) == 1 and len(exp) == 1:
        return {got[0], exp[0]} <= {"bool", "choice"}
    return False


def _case_passes(case) -> bool:
    spec = SchemaInferencer.from_prompt(case["use_case"])
    got_types = [q.type.value for q in spec.questions]
    binary = bool(case.get("binary"))
    if not _types_match(got_types, case["types"], binary):
        return False
    if "options" in case and len(got_types) == len(case["types"]):
        for idx_s, exp in case["options"].items():
            got = {_norm(o) for o in spec.questions[int(idx_s)].options}
            if binary and not got:
                continue
            if got != {_norm(o) for o in exp}:
                return False
    return True


def test_schema_benchmark_accuracy():
    cases = json.load(open(BENCH))["cases"]
    passed = sum(_case_passes(c) for c in cases)
    acc = passed / len(cases)
    assert acc >= THRESHOLD, f"schema benchmark {passed}/{len(cases)} = {acc:.1%} < {THRESHOLD:.0%}"
