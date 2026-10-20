"""Labeler: assign gold answers via multi-teacher consensus.

Each state is labeled independently by every configured teacher. For each
question the per-teacher answers are aggregated into a single gold answer plus
an **agreement score** in [0, 1]:

  - CHOICE / BOOL : majority (or unanimous) vote; agreement = winning fraction.
  - SCORE         : mean of numeric votes; agreement = 1 - normalized spread.

Low-agreement rows are the "labels != truth" risk we flagged. They are kept but
surfaced through `review_queue()` so a human can spot-check them before export.
For regulated use cases, route that queue to a reviewer and gate on it.

Teachers are pluggable. The default `heuristic` teacher is dependency-free and
deterministic, so the pipeline runs end-to-end with no API keys. Real LLM
teachers (`"gpt"`, `"bedrock:claude"`) load via the optional `[llm]` extra; if a
requested teacher can't be constructed we fall back to the heuristic so the run
never hard-fails on a missing credential.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .forge import DataForge
from .types import (
    DEFAULT_SCORE_RANGE,
    UNPARSEABLE,
    QuestionType,
    Question,
    Row,
    Spec,
    validate_answer,
)

# A teacher is anything that, given (state_text, question), returns an answer.
Teacher = Callable[[str, Question], Any]


class Labeler:
    """Consensus labeler over one or more teachers."""

    def __init__(
        self,
        teacher: Union[str, Sequence[str], Sequence[Teacher], None] = None,
        vote: str = "majority",
        strict: bool = True,
    ):
        """
        teacher : a teacher id, a list of ids, or a list of callables.
                  Defaults to a single `heuristic` teacher.
        vote    : "majority" (plurality wins) or "unanimous" (agreement only
                  counts a win when all teachers concur).
        strict  : when True (default), a NAMED non-heuristic teacher that cannot
                  be constructed (missing SDK, bad/absent credentials, unknown
                  id) raises a RuntimeError instead of silently falling back to
                  the keyword heuristic. This prevents the dangerous failure the
                  review flagged: asking for `openai:gpt-4o-mini` and quietly
                  getting heuristic labels that wear the LLM's name. Set
                  strict=False only for the no-key demo path, where the
                  heuristic stand-in is intended.
        """
        self.vote = vote
        self.strict = strict
        self._teachers: List[tuple] = self._resolve_teachers(teacher)

    # -- public API ----------------------------------------------------------

    def label(self, forge: DataForge, vote: Optional[str] = None) -> "LabeledRows":
        vote_mode = vote or self.vote
        spec = forge.spec
        rows: List[Row] = []
        for st in forge.states:
            row = self._label_state(st.text, spec, vote_mode)
            row.synthetic = st.synthetic
            rows.append(row)
        return LabeledRows(spec=spec, rows=rows)

    # -- labeling internals --------------------------------------------------

    def _label_state(self, text: str, spec: Spec, vote_mode: str) -> Row:
        gold: Dict[str, Any] = {}
        agreement: Dict[str, float] = {}
        votes: Dict[str, Dict[str, Any]] = {}

        for q in spec.questions:
            per_teacher: Dict[str, Any] = {}
            for name, fn in self._teachers:
                # Validate EVERY teacher's answer at this single boundary, so a
                # BYO callable obeys the same schema as a provider: anything
                # that isn't exactly one valid answer becomes UNPARSEABLE (an
                # abstention) rather than an ordinary gold label.
                per_teacher[name] = validate_answer(fn(text, q), q)
            answer, score = self._aggregate(q, per_teacher, vote_mode)
            gold[q.name] = answer
            agreement[q.name] = score
            votes[q.name] = per_teacher

        return Row(state=text, gold=gold, agreement=agreement, votes=votes)

    def _aggregate(self, q: Question, per_teacher: Dict[str, Any], vote_mode: str):
        all_values = list(per_teacher.values())
        if not all_values:
            return None, 0.0

        total = len(all_values)
        # An UNPARSEABLE vote is an abstention: it never wins, but it counts
        # against agreement so any row with an unvalidated answer is pushed
        # toward the review queue instead of quietly adopting a coerced label.
        n_unparseable = sum(1 for v in all_values if v is UNPARSEABLE)
        values = [v for v in all_values if v is not UNPARSEABLE]
        if not values:
            # Every teacher failed to produce a valid answer -> no gold, flag it.
            return None, 0.0

        if q.type is QuestionType.SCORE:
            nums = []
            for v in values:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    continue
            if not nums:
                return None, 0.0
            mean = statistics.fmean(nums)
            lo, hi = (q.score_range or DEFAULT_SCORE_RANGE)
            span = (hi - lo) or 1.0
            spread = (max(nums) - min(nums)) / span
            base = max(0.0, 1.0 - spread)
            # Penalize agreement by the fraction of teachers that abstained.
            score = base * (len(nums) / total)
            if vote_mode == "unanimous":
                # Unanimous requires EVERY teacher to have produced the same
                # valid number (no abstentions, zero spread); otherwise the row
                # is unresolved -> agreement 0 -> review queue.
                unanimous = len(nums) == total and max(nums) == min(nums)
                score = 1.0 if unanimous else 0.0
            return round(mean, 3), round(score, 3)

        # CHOICE / BOOL: categorical vote over the valid answers only. Tally by
        # a normalized KEY (case/space-insensitive) so "IT" and "it" count as
        # one, but keep the CANONICAL value a teacher actually returned as the
        # gold answer. Every value already passed validate_answer, so for CHOICE
        # it is the exact option string and for BOOL a real bool -- we must not
        # lowercase it, or export's exact-membership check would reject it and
        # silently drop the row.
        tally: Dict[Any, int] = {}
        canonical: Dict[Any, Any] = {}
        for v in values:
            key = self._norm(v)
            tally[key] = tally.get(key, 0) + 1
            canonical.setdefault(key, v)  # first-seen original form
        winner_key, count = max(tally.items(), key=lambda kv: kv[1])
        winner = canonical[winner_key]
        if vote_mode == "unanimous":
            # Confident only when ALL teachers concur on one valid answer: no
            # abstentions and a single distinct value. Any dissent/abstention
            # keeps the plurality winner as inspectable gold but forces
            # agreement to 0 so the row is routed to review.
            unanimous = len(values) == total and len(tally) == 1
            return winner, 1.0 if unanimous else 0.0
        # Majority: agreement is the winning fraction over ALL teachers
        # (abstentions included), so unparseable votes lower it.
        frac = count / total
        return winner, round(frac, 3)

    @staticmethod
    def _norm(v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    # -- teacher resolution --------------------------------------------------

    def _resolve_teachers(self, teacher) -> List[tuple]:
        if teacher is None:
            return [("heuristic", _HeuristicTeacher())]

        items = teacher if isinstance(teacher, (list, tuple)) else [teacher]
        resolved: List[tuple] = []
        for i, t in enumerate(items):
            if callable(t) and not isinstance(t, str):
                resolved.append((getattr(t, "name", f"teacher_{i}"), t))
                continue
            fn = self._load_named_teacher(str(t))
            resolved.append((str(t), fn))
        return resolved or [("heuristic", _HeuristicTeacher())]

    def _load_named_teacher(self, name: str) -> Teacher:
        # "heuristic" and seeded variants ("heuristic-b", ...) are the built-in
        # dependency-free teacher. These are never provider lookups, so they
        # never raise -- the seed just perturbs tie-breaking for demo diversity.
        if name == "heuristic" or name.startswith("heuristic-"):
            return _HeuristicTeacher(label=name)
        # Named non-heuristic teacher: try to construct the provider. In strict
        # mode, any failure (missing SDK/credentials, or an unrecognized id that
        # resolves to no provider) raises instead of silently returning the
        # heuristic. A silent fallback would mint a whole dataset of keyword
        # labels under the requested model's name.
        err: Optional[Exception] = None
        try:
            from .providers import get_provider

            provider = get_provider(name)
            if provider is not None:
                return provider.as_teacher()
        except Exception as e:  # SDK import error, bad creds, etc.
            err = e

        if self.strict:
            detail = f" ({err})" if err else ""
            raise RuntimeError(
                f"teacher {name!r} could not be constructed{detail}. Install the "
                f"'[llm]' extra and set the provider credentials (e.g. "
                f"OPENAI_API_KEY / ANTHROPIC_API_KEY), check the model id, or pass "
                f"strict=False to allow the heuristic stand-in."
            )
        return _HeuristicTeacher(label=name)  # non-strict: deterministic stand-in


class _HeuristicTeacher:
    """Deterministic, dependency-free teacher.

    Not a real model: it picks answers from simple cues so the pipeline is
    runnable and reproducible. For CHOICE it scores each option by substring /
    token overlap with the state; for BOOL it looks for negation/affirmation
    cues; for SCORE it maps text length / intensity words into the range.

    When several heuristic teachers are configured with different `label`
    seeds, they perturb their tie-breaking so consensus/agreement is non-trivial
    in demos.
    """

    def __init__(self, label: str = "heuristic"):
        self.name = label
        self._salt = sum(ord(c) for c in label)

    def __call__(self, text: str, q: Question) -> Any:
        low = text.lower()
        if q.type is QuestionType.BOOL:
            neg = any(w in low for w in ("not ", "no ", "never", "without", "deny"))
            pos = any(w in low for w in ("refund", "yes", "flag", "please", "want"))
            return bool(pos and not neg)

        if q.type is QuestionType.SCORE:
            lo, hi = q.score_range or DEFAULT_SCORE_RANGE
            intensity = sum(low.count(w) for w in ("urgent", "asap", "angry", "!", "immediately"))
            frac = min(1.0, (len(low) / 240.0) + 0.2 * intensity)
            val = lo + frac * (hi - lo)
            return round(val, 1)

        # CHOICE: best option by token overlap, salted tie-break
        if not q.options:
            return None
        tokens = set(re.findall(r"[a-z]+", low))
        best, best_score = q.options[0], -1
        for i, opt in enumerate(q.options):
            opt_tokens = set(re.findall(r"[a-z]+", opt.lower()))
            overlap = len(tokens & opt_tokens)
            direct = 1 if opt.lower() in low else 0
            score = overlap * 2 + direct * 3 + ((self._salt + i) % 2) * 0  # salt reserved
            if score > best_score:
                best, best_score = opt, score
        # If nothing matched, salt decides to introduce mild disagreement.
        if best_score <= 0:
            best = q.options[self._salt % len(q.options)]
        return best


class LabeledRows:
    """Labeled rows + the entry point into the quality gates.

    `.decontaminate(...)`, `.balance()`, `.confidence_filter(...)` live on the
    `Dataset` returned here; see `gates.py`. `LabeledRows` itself exposes the
    review queue and basic stats.
    """

    def __init__(self, spec: Spec, rows: List[Row]):
        self.spec = spec
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def review_queue(self, min_agreement: float = 0.7) -> List[Row]:
        """Rows whose weakest question agreement is below `min_agreement`.

        These are the "labels may not equal truth" cases to spot-check first.
        """
        return [r for r in self.rows if r.min_agreement < min_agreement]

    def agreement_summary(self) -> Dict[str, float]:
        """Mean agreement per question across all rows."""
        out: Dict[str, List[float]] = {q.name: [] for q in self.spec.questions}
        for r in self.rows:
            for name, score in r.agreement.items():
                out.setdefault(name, []).append(score)
        return {k: round(statistics.fmean(v), 3) if v else 0.0 for k, v in out.items()}

    # Delegate the quality gates to the Dataset wrapper.
    def _as_dataset(self):
        from .gates import Dataset

        return Dataset(spec=self.spec, rows=list(self.rows))

    def decontaminate(self, *args, **kwargs):
        return self._as_dataset().decontaminate(*args, **kwargs)

    def balance(self, *args, **kwargs):
        return self._as_dataset().balance(*args, **kwargs)

    def confidence_filter(self, *args, **kwargs):
        return self._as_dataset().confidence_filter(*args, **kwargs)

    def to_jsonl(self, *args, **kwargs):
        return self._as_dataset().to_jsonl(*args, **kwargs)
