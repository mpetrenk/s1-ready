"""Tests for the fail-closed / validate-or-flag labeling hardening.

These cover the two review-flagged weaknesses:
  1. a named non-heuristic teacher that can't be constructed must RAISE in
     strict mode, not silently become the keyword heuristic;
  2. a teacher answer that isn't a validated member of the option set (or a
     parseable bool / in-range score) must be flagged for review (low
     agreement -> review queue), not silently coerced to False/options[0].

Dependency-free: no API keys, no models.
"""

from __future__ import annotations

import pytest

from s1_ready import DataForge, Labeler
from s1_ready.providers import _coerce_answer
from s1_ready.types import UNPARSEABLE, Question, QuestionType, Row, Spec


BOOL_Q = Question(name="is_spam", type=QuestionType.BOOL, prompt="spam?")
CHOICE_Q = Question(
    name="route", type=QuestionType.CHOICE,
    options=["billing", "sales", "support"], prompt="route?",
)
SCORE_Q = Question(
    name="urgency", type=QuestionType.SCORE, score_range=[1, 5], prompt="urgency?",
)


# ---- 1. fail-closed teacher resolution -------------------------------------

def test_strict_unknown_teacher_raises():
    # An id that resolves to no provider must raise in strict mode rather than
    # quietly degrade to the heuristic. (Uses a name no provider recognizes, so
    # the result is deterministic regardless of which SDKs/keys are installed.)
    with pytest.raises(RuntimeError):
        Labeler(teacher="nonexistent-vendor:some-model")  # strict=True by default


def test_strict_bedrock_without_aws_raises():
    # A provider-shaped id whose SDK/creds are absent must also raise, not fall
    # back. boto3 or AWS creds are not available in the test env.
    with pytest.raises(RuntimeError):
        Labeler(teacher="bedrock:anthropic.claude-3-haiku-20240307-v1:0")


def test_strict_is_default():
    lab = Labeler(teacher="heuristic")
    assert lab.strict is True


def test_non_strict_falls_back_to_heuristic():
    # Opt-in escape hatch for the no-key demo path: no raise, heuristic stand-in.
    lab = Labeler(teacher="nonexistent-vendor:some-model", strict=False)
    assert len(lab._teachers) == 1
    name, fn = lab._teachers[0]
    assert name == "nonexistent-vendor:some-model"  # keeps the requested name
    # It still produces *a* label (from the heuristic), proving it fell back.
    assert fn("win a free prize now", BOOL_Q) in (True, False)


def test_heuristic_and_callables_still_work_in_strict():
    # Default strict mode must not break the dependency-free paths.
    Labeler()                      # default heuristic
    Labeler(teacher="heuristic")   # explicit heuristic
    Labeler(teacher=lambda text, q: "billing")  # BYO callable


# ---- 2. validate-or-flag coercion ------------------------------------------

def test_coerce_bool_valid():
    # Only an unambiguous single-token answer is accepted.
    assert _coerce_answer("true", BOOL_Q) is True
    assert _coerce_answer("True", BOOL_Q) is True
    assert _coerce_answer("yes", BOOL_Q) is True
    assert _coerce_answer("false", BOOL_Q) is False
    assert _coerce_answer("no", BOOL_Q) is False


def test_coerce_bool_ambiguous_and_negated_are_unparseable():
    # Exactly the reviewer's cases: loose token/substring search used to turn
    # these into confident (wrong) labels.
    assert _coerce_answer("not true", BOOL_Q) is UNPARSEABLE
    assert _coerce_answer("false, not true", BOOL_Q) is UNPARSEABLE
    assert _coerce_answer("maybe?", BOOL_Q) is UNPARSEABLE
    assert _coerce_answer("I cannot determine that", BOOL_Q) is UNPARSEABLE


def test_coerce_choice_exact_only():
    assert _coerce_answer("billing", CHOICE_Q) == "billing"
    assert _coerce_answer("  SUPPORT ", CHOICE_Q) == "support"  # case/space ok
    # Reviewer's cases: substring / multi-option must NOT resolve.
    assert _coerce_answer("unsupported", CHOICE_Q) is UNPARSEABLE
    assert _coerce_answer("billing or sales; cannot decide", CHOICE_Q) is UNPARSEABLE
    assert _coerce_answer("escalate to management", CHOICE_Q) is UNPARSEABLE


def test_coerce_score_exact_in_range():
    assert _coerce_answer("3", SCORE_Q) == 3.0
    assert _coerce_answer(" 5 ", SCORE_Q) == 5.0
    # No longer digs a number out of prose, and rejects out-of-range.
    assert _coerce_answer("urgency is 5", SCORE_Q) is UNPARSEABLE
    assert _coerce_answer("3 or 4", SCORE_Q) is UNPARSEABLE
    assert _coerce_answer("9", SCORE_Q) is UNPARSEABLE
    assert _coerce_answer("not sure", SCORE_Q) is UNPARSEABLE


# ---- 2b. callable teachers obey the same validation ------------------------

def test_callable_invalid_answer_is_flagged_not_labeled():
    # A callable returning "maybe" for a bool used to become gold="maybe" at
    # agreement 1.0. Now it abstains and the row has no gold + lands in review.
    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    lab = Labeler(teacher=lambda text, q: "maybe")
    labeled = lab.label(DataForge(spec).ingest(["hello"]))
    row = labeled.rows[0]
    assert row.gold["is_spam"] is None
    assert row.agreement["is_spam"] == 0.0
    assert row in labeled.review_queue(min_agreement=0.7)


def test_callable_out_of_set_choice_is_flagged():
    spec = Spec(use_case="route?", questions=[CHOICE_Q])
    lab = Labeler(teacher=lambda text, q: "unsupported")
    labeled = lab.label(DataForge(spec).ingest(["hello"]))
    assert labeled.rows[0].gold["route"] is None


# ---- 3. unparseable votes flow to the review queue -------------------------

def _teacher_returning(value):
    def fn(text, q):
        return value
    fn.name = f"stub_{value}"
    return fn


def test_unparseable_vote_lowers_agreement_and_flags_row():
    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    # Two teachers: one says True, one abstains (UNPARSEABLE).
    lab = Labeler(teacher=[_teacher_returning(True), _teacher_returning(UNPARSEABLE)])
    forge = DataForge(spec).ingest(["win a free prize"])
    labeled = lab.label(forge)
    row = labeled.rows[0]
    # Gold is the one valid vote; agreement is 1/2 because the abstention counts.
    assert row.gold["is_spam"] is True
    assert row.agreement["is_spam"] == 0.5
    # A <0.7 agreement row must surface in the review queue.
    assert row in labeled.review_queue(min_agreement=0.7)


def test_all_unparseable_gives_no_gold():
    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    lab = Labeler(teacher=[_teacher_returning(UNPARSEABLE)])
    forge = DataForge(spec).ingest(["???"])
    labeled = lab.label(forge)
    row = labeled.rows[0]
    assert row.gold["is_spam"] is None
    assert row.agreement["is_spam"] == 0.0
    assert row in labeled.review_queue(min_agreement=0.7)


# ---- 4. vote="unanimous" has distinct behavior -----------------------------

def test_unanimous_requires_all_agree():
    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    # Two teachers that disagree. Majority -> agreement 0.5; unanimous -> 0.0.
    teachers = [_teacher_returning(True), _teacher_returning(False)]
    maj = Labeler(teacher=teachers, vote="majority").label(
        DataForge(spec).ingest(["x"])).rows[0]
    uni = Labeler(teacher=teachers, vote="unanimous").label(
        DataForge(spec).ingest(["x"])).rows[0]
    assert maj.agreement["is_spam"] == 0.5
    assert uni.agreement["is_spam"] == 0.0  # distinct from majority


def test_unanimous_full_agreement_scores_one():
    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    teachers = [_teacher_returning(True), _teacher_returning(True)]
    uni = Labeler(teacher=teachers, vote="unanimous").label(
        DataForge(spec).ingest(["x"])).rows[0]
    assert uni.gold["is_spam"] is True
    assert uni.agreement["is_spam"] == 1.0


# ---- 5. training-export gate excludes invalid-gold rows --------------------

def test_training_export_excludes_invalid_gold_rows():
    from s1_ready.gates import Dataset
    from s1_ready.export import to_records, row_has_valid_gold

    spec = Spec(use_case="spam?", questions=[BOOL_Q])
    good = Row(state="clearly spam", gold={"is_spam": True}, agreement={"is_spam": 1.0})
    bad = Row(state="???", gold={"is_spam": None}, agreement={"is_spam": 0.0})
    ds = Dataset(spec=spec, rows=[good, bad])

    assert row_has_valid_gold(ds, good) is True
    assert row_has_valid_gold(ds, bad) is False

    # Training formats drop the invalid row...
    for fmt in ("unsloth", "conversational", "raw"):
        recs = to_records(ds, fmt=fmt)
        assert len(recs) == 1, fmt
    # ...while the auditable typed-decisions format keeps everything.
    assert len(to_records(ds, fmt="typed-decisions")) == 2
