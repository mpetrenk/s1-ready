"""Core data types shared across the s1-ready pipeline.

A System 1 (decision-model) training row is a `(state, questions, gold)` triple:

    state     - the raw input text to judge
    questions - the decision schema: for each question, a type + option set
    gold      - the chosen answer(s), one per question, plus confidence/agreement

These are plain dataclasses with no third-party dependencies so the core
pipeline runs on the stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class QuestionType(str, Enum):
    """The three decision shapes a System 1 model answers.

    CHOICE - pick exactly one option from a fixed set (e.g. route-to-team).
    BOOL   - a yes/no flag (e.g. is-refund). (The brainstorm's "noul" was a
             typo for a boolean/null flag; we model it as BOOL.)
    SCORE  - an ordinal/numeric rating within a bounded range.
    """

    CHOICE = "choice"
    BOOL = "bool"
    SCORE = "score"


@dataclass
class Question:
    """One decision to make about a state."""

    name: str
    type: QuestionType
    options: List[str] = field(default_factory=list)
    # For SCORE questions: inclusive numeric bounds.
    score_range: Optional[List[float]] = None
    prompt: str = ""

    def is_valid_answer(self, answer: Any) -> bool:
        if self.type is QuestionType.CHOICE:
            return answer in self.options
        if self.type is QuestionType.BOOL:
            return isinstance(answer, bool) or answer in (
                "true", "false", "yes", "no", True, False,
            )
        if self.type is QuestionType.SCORE:
            try:
                val = float(answer)
            except (TypeError, ValueError):
                return False
            lo, hi = self.score_range or [0.0, 1.0]
            return lo <= val <= hi
        return False

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "type": self.type.value}
        if self.type is QuestionType.CHOICE:
            d["options"] = list(self.options)
        if self.type is QuestionType.SCORE:
            d["score_range"] = self.score_range or [0.0, 1.0]
        if self.prompt:
            d["prompt"] = self.prompt
        return d


@dataclass
class Spec:
    """The inferred decision schema for a use case.

    A `use_case` sentence maps to one or more `questions`, each with its own
    option set. This is the artifact that turns unstructured text into
    something a decision model can be trained to answer.
    """

    use_case: str
    questions: List[Question]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_case": self.use_case,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class Row:
    """A single (state, gold) example aligned to a Spec."""

    state: str
    # question-name -> chosen answer
    gold: Dict[str, Any] = field(default_factory=dict)
    # question-name -> agreement score in [0, 1] from teacher consensus
    agreement: Dict[str, float] = field(default_factory=dict)
    # per-question votes (teacher -> answer), kept for the review queue
    votes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def min_agreement(self) -> float:
        return min(self.agreement.values()) if self.agreement else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
