"""DataForge: assemble the pool of states to be labeled.

Responsibilities:

  - ingest()    : load real, unlabeled states from a corpus (file or iterable).
  - synthesize(): generate in-distribution filler states to cover sparse
                  option values, so the labeled distribution stays balanced and
                  the resulting model's confidence stays calibrated.
  - balance()   : report / drive balancing against a target per-option count.

The forge holds *states only* (plus a `synthetic` flag). Labeling happens later
in `Labeler`, which assigns the gold answers. Synthesis here is template-based
and dependency-free; when an LLM teacher is configured, `synthesize(use_llm=True)`
can delegate to it for more natural fillers.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .types import QuestionType, Spec


@dataclass
class ForgeState:
    """A single state in the pool, before labeling."""

    text: str
    synthetic: bool = False
    # Optional hint: for synthesized states, which option they were seeded to
    # cover (question-name -> option). Used by balance() diagnostics.
    seed: Dict[str, str] = field(default_factory=dict)


class DataForge:
    """Holds the pool of states for a given decision `Spec`."""

    def __init__(self, spec: Spec, seed: int = 0):
        self.spec = spec
        self.states: List[ForgeState] = []
        self._rng = random.Random(seed)

    # -- ingest --------------------------------------------------------------

    def ingest(self, source: Union[str, Iterable[str]]) -> "DataForge":
        """Load unlabeled states.

        `source` may be a path to a file (one state per non-empty line), a raw
        multi-line string, or an iterable of strings.
        """
        for text in self._read_source(source):
            self.states.append(ForgeState(text=text, synthetic=False))
        return self

    # -- synthesize ----------------------------------------------------------

    def synthesize(
        self,
        n: int = 0,
        balance: bool = True,
        use_llm: bool = False,
        teacher: Optional[str] = None,
    ) -> "DataForge":
        """Generate up to `n` synthetic states.

        When `balance=True`, synthesis is steered toward the CHOICE options that
        are under-represented in the current pool (estimated by keyword
        presence), so sparse buckets get filled first. The remaining budget is
        spread across all options round-robin.
        """
        if n <= 0:
            return self

        choice_qs = [q for q in self.spec.questions if q.type is QuestionType.CHOICE]
        if not choice_qs:
            # No categorical axis to balance; emit generic paraphrase fillers.
            for _ in range(n):
                self.states.append(self._generic_filler())
            return self

        targets = self._balance_targets(choice_qs, n) if balance else None

        if use_llm:
            llm_states = self._synthesize_with_llm(n, teacher, targets)
            if llm_states is not None:
                self.states.extend(llm_states)
                return self
            # fall through to template synthesis if LLM unavailable

        self._synthesize_templated(choice_qs, n, targets)
        return self

    def _balance_targets(
        self, choice_qs: Sequence, n: int
    ) -> Dict[str, Dict[str, int]]:
        """Return question-name -> {option -> how many fillers to seed}.

        Fills the gap between each option's current (estimated) count and the
        max option count, then distributes any leftover budget evenly.
        """
        targets: Dict[str, Dict[str, int]] = {}
        remaining = n
        for q in choice_qs:
            counts = self._estimate_option_counts(q)
            peak = max(counts.values()) if counts else 0
            gaps = {opt: max(0, peak - counts.get(opt, 0)) for opt in q.options}
            targets[q.name] = gaps
            remaining -= sum(gaps.values())

        if remaining > 0 and choice_qs:
            q = choice_qs[0]
            opts = q.options or ["option_1"]
            for i in range(remaining):
                opt = opts[i % len(opts)]
                targets[q.name][opt] = targets[q.name].get(opt, 0) + 1
        return targets

    def _estimate_option_counts(self, q) -> Dict[str, int]:
        """Cheap unlabeled estimate: count states whose text mentions the option."""
        counts = Counter()
        for st in self.states:
            low = st.text.lower()
            for opt in q.options:
                if opt.lower() in low:
                    counts[opt] += 1
        return dict(counts)

    def _synthesize_templated(self, choice_qs, n: int, targets) -> None:
        made = 0
        if targets:
            for q in choice_qs:
                for opt, cnt in targets.get(q.name, {}).items():
                    for _ in range(cnt):
                        if made >= n:
                            return
                        self.states.append(self._seeded_filler(q.name, opt))
                        made += 1
        # top up any shortfall with generic fillers
        while made < n:
            self.states.append(self._generic_filler())
            made += 1

    def _seeded_filler(self, qname: str, option: str) -> ForgeState:
        templates = [
            "This case clearly relates to {opt}.",
            "Customer message indicating a {opt} situation.",
            "Example state seeded for the '{opt}' bucket.",
            "A representative {opt} scenario for calibration.",
        ]
        tmpl = self._rng.choice(templates)
        return ForgeState(
            text=tmpl.format(opt=option),
            synthetic=True,
            seed={qname: option},
        )

    def _generic_filler(self) -> ForgeState:
        return ForgeState(
            text="Synthetic in-distribution filler state for calibration.",
            synthetic=True,
        )

    def _synthesize_with_llm(self, n, teacher, targets) -> Optional[List[ForgeState]]:
        try:
            from .providers import get_provider
        except Exception:
            return None
        provider = get_provider(teacher) if teacher else None
        if provider is None:
            return None
        texts = provider.synthesize_states(self.spec, n, targets)
        return [ForgeState(text=t, synthetic=True) for t in texts]

    # -- balance diagnostics -------------------------------------------------

    def balance(self) -> Dict[str, Dict[str, int]]:
        """Return current per-option occurrence estimates per CHOICE question.

        Diagnostic only (labels don't exist yet). Useful to confirm synthesis
        closed the skew before handing the pool to the Labeler.
        """
        report: Dict[str, Dict[str, int]] = {}
        for q in self.spec.questions:
            if q.type is QuestionType.CHOICE:
                report[q.name] = self._estimate_option_counts(q)
        return report

    # -- misc ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.states)

    @property
    def texts(self) -> List[str]:
        return [s.text for s in self.states]

    @staticmethod
    def _read_source(source: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(source, str):
            # Treat as a path if it looks like one and exists; else as raw text.
            looks_like_path = (
                "\n" not in source and len(source) < 4096 and re.search(r"\.\w+$", source)
            )
            if looks_like_path:
                try:
                    with open(source, "r", encoding="utf-8") as fh:
                        return [ln.strip() for ln in fh if ln.strip()]
                except OSError:
                    pass
            return [ln.strip() for ln in source.splitlines() if ln.strip()]
        if hasattr(source, "read"):
            return [ln.strip() for ln in source.read().splitlines() if ln.strip()]
        return [str(s).strip() for s in source if str(s).strip()]
