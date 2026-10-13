"""Core pipeline tests for s1-ready.

Dependency-free: uses the built-in heuristic teacher, so these run anywhere
`pytest` is installed, no API keys or models required.

    pip install -e ".[dev]"
    pytest
"""

from __future__ import annotations

import json

from s1_ready import SchemaInferencer, DataForge, Labeler
from s1_ready.types import QuestionType, Question, Spec, Row
from s1_ready.gates import Dataset


# ---- schema inference ------------------------------------------------------

def test_schema_single_choice_with_inline_options():
    spec = SchemaInferencer.from_prompt(
        "Route tickets to billing, sales, or support"
    )
    assert len(spec.questions) == 1
    q = spec.questions[0]
    assert q.type is QuestionType.CHOICE
    assert set(q.options) == {"billing", "sales", "support"}


def test_schema_splits_choice_and_bool():
    spec = SchemaInferencer.from_prompt(
        "Route tickets to billing, sales, or support, and flag refunds"
    )
    types = {q.type for q in spec.questions}
    assert QuestionType.CHOICE in types
    assert QuestionType.BOOL in types
    # the choice keeps its 3 options, not shattered into separate questions
    choice = next(q for q in spec.questions if q.type is QuestionType.CHOICE)
    assert set(choice.options) == {"billing", "sales", "support"}


def test_schema_score_detected():
    spec = SchemaInferencer.from_prompt("Rate urgency 1-5")
    assert spec.questions[0].type is QuestionType.SCORE


def test_llm_teacher_is_vendor_neutral_callable():
    # The LLM path must accept any callable fn(prompt)->str, with no OpenAI SDK,
    # key, or network. This guards against the teacher becoming OpenAI-only.
    def fake_llm(prompt: str) -> str:
        return ('{"questions":[{"name":"route","type":"choice",'
                '"options":["a","b","c"]}]}')

    spec = SchemaInferencer.from_prompt("Route to a, b, or c", teacher=fake_llm)
    assert len(spec.questions) == 1
    q = spec.questions[0]
    assert q.type is QuestionType.CHOICE
    assert q.options == ["a", "b", "c"]


def test_anthropic_id_routes_to_direct_provider(monkeypatch):
    # "anthropic:<model>" must resolve to the native Anthropic provider (not
    # Bedrock) when ANTHROPIC_API_KEY is present. No network call is made.
    import importlib.util
    if importlib.util.find_spec("anthropic") is None:
        import pytest
        pytest.skip("anthropic SDK not installed")
    from s1_ready.providers import get_provider, _AnthropicProvider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-not-real")
    p = get_provider("anthropic:claude-3-5-haiku-latest")
    assert isinstance(p, _AnthropicProvider)


def test_anthropic_id_without_key_falls_back(monkeypatch):
    from s1_ready.providers import get_provider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_provider("anthropic:claude-3-5-haiku-latest") is None


def test_llm_teacher_bad_output_falls_back_to_heuristic():
    # A teacher that returns unusable output must not crash -> heuristic fallback.
    spec = SchemaInferencer.from_prompt(
        "Route tickets to billing, sales, or support",
        teacher=lambda prompt: "sorry, I cannot help with that",
    )
    # heuristic still produces the correct choice schema
    assert spec.questions[0].type is QuestionType.CHOICE
    assert set(spec.questions[0].options) == {"billing", "sales", "support"}


def test_inline_options_beat_keyword_cue():
    # "sentiment" is a SCORE cue word, but the explicit option list must win:
    # this clause is a CHOICE. Regression test for that misclassification.
    spec = SchemaInferencer.from_prompt(
        "Classify the review sentiment as positive, negative, or neutral, and rate severity 1-5"
    )
    assert len(spec.questions) == 2
    by_type = {q.type for q in spec.questions}
    assert QuestionType.CHOICE in by_type
    assert QuestionType.SCORE in by_type
    choice = next(q for q in spec.questions if q.type is QuestionType.CHOICE)
    assert set(choice.options) == {"positive", "negative", "neutral"}


# ---- forge -----------------------------------------------------------------

def test_forge_ingest_and_synthesize():
    spec = SchemaInferencer.from_prompt("Route to billing, sales, or support")
    forge = DataForge(spec, seed=0).ingest("needs a refund\nsales question\nhelp please")
    assert len(forge) == 3
    forge.synthesize(n=6, balance=True)
    assert len(forge) == 9
    assert sum(s.synthetic for s in forge.states) == 6


# ---- labeler ---------------------------------------------------------------

def test_labeler_produces_gold_and_agreement():
    spec = SchemaInferencer.from_prompt("Decide whether a message is spam")
    forge = DataForge(spec).ingest("win a free prize now\nsee you at lunch")
    rows = Labeler(teacher=["heuristic", "heuristic-b"]).label(forge)
    assert len(rows) == 2
    for r in rows.rows:
        assert spec.questions[0].name in r.gold
        assert 0.0 <= r.min_agreement <= 1.0


def test_review_queue_flags_low_agreement():
    spec = SchemaInferencer.from_prompt("Decide whether a message is spam")
    forge = DataForge(spec).ingest("ambiguous text")
    rows = Labeler(teacher=["heuristic", "heuristic-b", "heuristic-c"]).label(forge)
    # review_queue returns a subset; just assert it runs and respects threshold
    q = rows.review_queue(min_agreement=1.1)  # impossible threshold -> all flagged
    assert len(q) == len(rows.rows)


# ---- gates -----------------------------------------------------------------

def _toy_dataset():
    spec = Spec(
        use_case="is it spam",
        questions=[Question(name="is_spam", type=QuestionType.BOOL, prompt="spam?")],
    )
    rows = [
        Row(state="win a prize", gold={"is_spam": True}, agreement={"is_spam": 1.0}),
        Row(state="win a prize", gold={"is_spam": True}, agreement={"is_spam": 1.0}),
        Row(state="lunch at noon", gold={"is_spam": False}, agreement={"is_spam": 0.5}),
    ]
    return Dataset(spec=spec, rows=rows)


def test_confidence_filter():
    ds = _toy_dataset()
    ds.confidence_filter(min=0.7)
    assert len(ds) == 2  # the 0.5-agreement row dropped


def test_decontaminate_removes_overlap():
    ds = _toy_dataset()
    before = len(ds)
    ds.decontaminate(["win a prize"])  # near-dup of two training rows
    assert len(ds) < before


def test_split_is_disjoint_and_deterministic():
    spec = Spec(use_case="x", questions=[Question(name="q", type=QuestionType.BOOL)])
    rows = [Row(state=f"msg {i}", gold={"q": i % 2 == 0}) for i in range(50)]
    ds = Dataset(spec=spec, rows=list(rows))
    ds.split(test_frac=0.3, seed=1)
    train_states = {r.state for r in ds.rows}
    test_states = {r.state for r in (ds.test_rows or [])}
    assert train_states.isdisjoint(test_states)
    assert len(train_states) + len(test_states) == 50


# ---- export ----------------------------------------------------------------

def test_export_typed_decisions(tmp_path):
    ds = _toy_dataset()
    out = tmp_path / "d.jsonl"
    ds.to_jsonl(str(out), fmt="typed-decisions")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert set(rec) == {"state", "questions", "gold", "agreement", "synthetic"}
    assert (out.parent / "d.jsonl.spec.json").exists()


def test_export_conversational_shape(tmp_path):
    ds = _toy_dataset()
    out = tmp_path / "c.jsonl"
    ds.to_jsonl(str(out), fmt="conversational")
    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert list(rec) == ["conversations"]
    assert [m["role"] for m in rec["conversations"]] == ["user", "assistant"]


def test_export_unsloth_shape_and_helper(tmp_path):
    ds = _toy_dataset()
    out = tmp_path / "u.jsonl"
    ds.to_jsonl(str(out), fmt="unsloth", emit_helper=True)
    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert set(rec) == {"instruction", "input", "output"}
    assert (tmp_path / "u.jsonl.train.py").exists()


def test_unknown_format_raises(tmp_path):
    ds = _toy_dataset()
    import pytest
    with pytest.raises(ValueError):
        ds.to_jsonl(str(tmp_path / "x.jsonl"), fmt="nope")
