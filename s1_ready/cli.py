"""Command-line entry point: `s1-ready` (alias `s1ready`).

Runs the full pipeline end-to-end:

    s1-ready prepare \
        --use-case "Route support tickets to a team and flag refunds" \
        --corpus tickets.txt \
        --synthesize 500 \
        --teacher heuristic \
        --min-agreement 0.7 \
        --out train.jsonl

Steps: infer schema -> ingest corpus -> synthesize fillers -> consensus label
-> split/decontaminate -> balance -> confidence filter -> export JSONL.

Also:
    s1-ready schema --use-case "..." [--corpus FILE]   # print inferred schema
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .forge import DataForge
from .labeler import Labeler
from .schema import SchemaInferencer


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 1
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="s1-ready",
        description="Turn any data with a decision attached into a System 1-ready dataset.",
    )
    p.add_argument("--version", action="version", version=f"s1-ready {__version__}")
    sub = p.add_subparsers(dest="cmd")

    # prepare
    prep = sub.add_parser("prepare", help="run the full pipeline and export JSONL")
    prep.add_argument("--use-case", required=True, help="plain-English decision to prepare for")
    prep.add_argument("--corpus", help="path to unlabeled states (one per line)")
    prep.add_argument("--synthesize", type=int, default=0, help="N synthetic fillers to add")
    prep.add_argument(
        "--teacher",
        default="heuristic",
        help="comma-separated teacher ids (e.g. 'heuristic' or 'gpt,bedrock:claude')",
    )
    prep.add_argument("--vote", default="majority", choices=["majority", "unanimous"])
    prep.add_argument("--min-agreement", type=float, default=0.0, help="drop rows below this agreement")
    prep.add_argument("--test-frac", type=float, default=0.0, help="hold out this fraction as test")
    prep.add_argument("--balance", action="store_true", help="down-sample to the minority bucket")
    prep.add_argument(
        "--format",
        default="typed-decisions",
        choices=["typed-decisions", "unsloth", "conversational", "raw"],
    )
    prep.add_argument(
        "--emit-helper",
        action="store_true",
        help="also write a <out>.train.py helper (unsloth/conversational formats only)",
    )
    prep.add_argument("--out", required=True, help="output JSONL path")
    prep.add_argument("--seed", type=int, default=0)
    prep.set_defaults(func=_cmd_prepare)

    # schema (inspect only)
    sc = sub.add_parser("schema", help="infer and print the decision schema only")
    sc.add_argument("--use-case", required=True)
    sc.add_argument("--corpus", help="sample states to mine options from")
    sc.add_argument("--teacher", default=None)
    sc.set_defaults(func=_cmd_schema)

    return p


def _cmd_schema(args) -> int:
    samples = _load_lines(args.corpus) if args.corpus else None
    spec = SchemaInferencer.from_prompt(
        args.use_case, sample_states=samples, teacher=args.teacher
    )
    print(json.dumps(spec.to_dict(), indent=2))
    return 0


def _cmd_prepare(args) -> int:
    teachers = [t.strip() for t in args.teacher.split(",") if t.strip()]
    samples = _load_lines(args.corpus) if args.corpus else None

    # 1. schema
    spec = SchemaInferencer.from_prompt(
        args.use_case, sample_states=samples, teacher=teachers[0] if teachers else None
    )
    _log(f"schema: {len(spec.questions)} question(s) -> "
         f"{', '.join(q.name + ':' + q.type.value for q in spec.questions)}")

    # 2. forge: ingest + synthesize
    forge = DataForge(spec, seed=args.seed)
    if args.corpus:
        forge.ingest(args.corpus)
    _log(f"ingested {len(forge)} state(s) from corpus")
    if args.synthesize > 0:
        forge.synthesize(n=args.synthesize, balance=True)
        _log(f"pool now {len(forge)} state(s) after synthesis")

    if len(forge) == 0:
        _log("no states to label; provide --corpus and/or --synthesize", err=True)
        return 2

    # 3. consensus label
    rows = Labeler(teacher=teachers, vote=args.vote).label(forge)
    summary = rows.agreement_summary()
    _log(f"labeled {len(rows)} row(s); mean agreement {summary}")
    queue = rows.review_queue(min_agreement=max(args.min_agreement, 0.7))
    if queue:
        _log(f"review queue: {len(queue)} low-agreement row(s) flagged for human spot-check")

    # 4. gates: split/decontaminate -> balance -> confidence filter
    ds = rows._as_dataset()
    if args.test_frac > 0:
        ds.split(test_frac=args.test_frac, seed=args.seed)
        held = len(ds.test_rows or [])
        ds.decontaminate(ds.test_rows or [])
        _log(f"held out {held} test row(s); train decontaminated to {len(ds)} row(s)")
    if args.balance:
        before = len(ds)
        ds.balance()
        _log(f"balanced: {before} -> {len(ds)} row(s)")
    if args.min_agreement > 0:
        before = len(ds)
        ds.confidence_filter(min=args.min_agreement)
        _log(f"confidence filter >= {args.min_agreement}: {before} -> {len(ds)} row(s)")

    # 5. export
    emit_helper = args.emit_helper
    if emit_helper and args.format not in ("unsloth", "conversational"):
        _log(
            f"--emit-helper ignored: no train helper for format {args.format!r} "
            "(use unsloth or conversational)",
            err=True,
        )
        emit_helper = False

    out = ds.to_jsonl(args.out, fmt=args.format, emit_helper=emit_helper)
    _log(f"wrote {len(ds)} row(s) -> {out}  (schema sidecar: {out}.spec.json)")
    if emit_helper:
        _log(f"train helper: {out}.train.py")
    print(out)
    return 0


# ---- helpers ---------------------------------------------------------------


def _load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _log(msg: str, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stderr
    print(f"[s1-ready] {msg}", file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
