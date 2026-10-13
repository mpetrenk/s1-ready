"""Quality gates + the Dataset container.

`Dataset` wraps labeled `Row`s and exposes the gates that make a corpus
*System 1-ready*. Every gate returns a `Dataset` so they chain:

    ds = (rows
          .decontaminate(test_set)      # drop train rows that leak into test
          .balance()                    # cap over-represented option buckets
          .confidence_filter(min=0.7))  # drop low-agreement rows
    ds.to_jsonl("train.jsonl")

Gates implemented:

  decontaminate(test_set)  guarantee train/test separation (the thing Unsloth
                           and Strands both call out). Removes any training row
                           whose state near-duplicates a test state.
  split(test_frac)         deterministic train/test split with no leakage.
  balance(max_per_option)  cap the majority option(s) so the label distribution
                           isn't skewed (keeps confidence calibrated).
  confidence_filter(min)   drop rows whose weakest question agreement < min.

Decontamination uses a dependency-free normalized-shingle signature, so no
third-party similarity library is required.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .types import QuestionType, Row, Spec


@dataclass
class Dataset:
    """Labeled rows aligned to a `Spec`, with chainable quality gates."""

    spec: Spec
    rows: List[Row] = field(default_factory=list)
    # Populated by split(); None until then.
    test_rows: Optional[List[Row]] = None

    def __len__(self) -> int:
        return len(self.rows)

    # -- decontamination -----------------------------------------------------

    def decontaminate(
        self,
        test_set: Union["Dataset", Iterable[str], Iterable[Row], None],
        threshold: float = 0.8,
    ) -> "Dataset":
        """Drop training rows that overlap the given test set.

        `test_set` may be a Dataset, an iterable of Rows, or an iterable of raw
        state strings. Overlap is measured by Jaccard similarity of word
        shingles; rows at/above `threshold` are removed from the training rows.
        """
        test_texts = self._coerce_texts(test_set)
        test_sigs = [self._shingles(t) for t in test_texts]
        if not test_sigs:
            return self

        kept: List[Row] = []
        for r in self.rows:
            sig = self._shingles(r.state)
            if any(self._jaccard(sig, ts) >= threshold for ts in test_sigs):
                continue  # leaked into test -> drop
            kept.append(r)
        self.rows = kept
        return self

    # -- split ---------------------------------------------------------------

    def split(self, test_frac: float = 0.2, seed: int = 0) -> "Dataset":
        """Deterministic train/test split with no cross-contamination.

        Rows are assigned by a stable hash of their state so the same row always
        lands in the same side regardless of ordering. Near-duplicate states
        hash to the same bucket side, preventing leakage across the boundary.
        Populates `self.test_rows`; `self.rows` becomes the training side.
        """
        test_frac = min(max(test_frac, 0.0), 1.0)
        cutoff = int(test_frac * 1000)
        train: List[Row] = []
        test: List[Row] = []
        for r in self.rows:
            bucket = self._stable_bucket(r.state, seed) % 1000
            (test if bucket < cutoff else train).append(r)
        self.rows = train
        self.test_rows = test
        return self

    # -- balance -------------------------------------------------------------

    def balance(self, max_per_option: Optional[int] = None) -> "Dataset":
        """Cap over-represented option buckets per CHOICE/BOOL question.

        Without `max_per_option`, caps every bucket at the size of the smallest
        non-empty bucket (full down-sampling to the minority class). Rows are
        kept in order; synthetic rows are dropped before real ones when a bucket
        must shrink, so real data is preferred.
        """
        choice_qs = [
            q for q in self.spec.questions
            if q.type in (QuestionType.CHOICE, QuestionType.BOOL)
        ]
        if not choice_qs:
            return self

        # Balance against the first categorical question (the primary label).
        q = choice_qs[0]
        buckets: Dict[object, List[Row]] = defaultdict(list)
        for r in self.rows:
            buckets[self._key(r.gold.get(q.name))].append(r)

        if not buckets:
            return self

        if max_per_option is None:
            max_per_option = min(len(v) for v in buckets.values())

        kept: List[Row] = []
        for _, group in buckets.items():
            # prefer real rows: real first, then synthetic
            group_sorted = sorted(group, key=lambda r: r.synthetic)
            kept.extend(group_sorted[:max_per_option])
        # restore original order for determinism of downstream exports
        order = {id(r): i for i, r in enumerate(self.rows)}
        kept.sort(key=lambda r: order.get(id(r), 0))
        self.rows = kept
        return self

    # -- confidence filter ---------------------------------------------------

    def confidence_filter(self, min: float = 0.7) -> "Dataset":
        """Drop rows whose weakest per-question agreement is below `min`."""
        self.rows = [r for r in self.rows if r.min_agreement >= min]
        return self

    # -- diagnostics ---------------------------------------------------------

    def distribution(self) -> Dict[str, Dict[str, int]]:
        """Gold-label distribution per CHOICE/BOOL question."""
        out: Dict[str, Dict[str, int]] = {}
        for q in self.spec.questions:
            if q.type in (QuestionType.CHOICE, QuestionType.BOOL):
                counts: Dict[str, int] = defaultdict(int)
                for r in self.rows:
                    counts[str(r.gold.get(q.name))] += 1
                out[q.name] = dict(counts)
        return out

    # -- export (thin pass-through to export.py) -----------------------------

    def to_jsonl(
        self, path: str, fmt: str = "typed-decisions", emit_helper: bool = False
    ) -> str:
        from .export import write_jsonl

        return write_jsonl(self, path, fmt=fmt, emit_helper=emit_helper)

    def to_unsloth(self, path: str, emit_helper: bool = True) -> str:
        return self.to_jsonl(path, fmt="unsloth", emit_helper=emit_helper)

    def to_conversational(self, path: str, emit_helper: bool = True) -> str:
        return self.to_jsonl(path, fmt="conversational", emit_helper=emit_helper)

    def to_records(self, fmt: str = "typed-decisions") -> List[dict]:
        from .export import to_records

        return to_records(self, fmt=fmt)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _coerce_texts(test_set) -> List[str]:
        if test_set is None:
            return []
        if isinstance(test_set, Dataset):
            return [r.state for r in (test_set.rows + (test_set.test_rows or []))]
        texts: List[str] = []
        for item in test_set:
            if isinstance(item, Row):
                texts.append(item.state)
            else:
                texts.append(str(item))
        return texts

    @staticmethod
    def _shingles(text: str, k: int = 3) -> frozenset:
        words = re.findall(r"[a-z0-9]+", text.lower())
        if len(words) < k:
            return frozenset(words)
        return frozenset(
            " ".join(words[i : i + k]) for i in range(len(words) - k + 1)
        )

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def _stable_bucket(text: str, seed: int) -> int:
        norm = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        digest = hashlib.sha1(f"{seed}:{norm}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _key(value) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value
