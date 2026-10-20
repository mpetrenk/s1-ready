"""Optional LLM teacher providers.

The core pipeline never requires these. `get_provider(name)` returns a provider
instance when the backing SDK + credentials are available, or `None` so callers
transparently fall back to the dependency-free heuristic.

The prompts and parsing here are model-agnostic: plain instruction text and
tolerant JSON/answer extraction, with no vendor-specific phrasing. Any
instruction-following LLM can serve as the teacher. Only `_complete` (one string
in, one string out) is provider-specific.

Supported ids:
    "gpt", "openai", "openai:<model>"   -> OpenAI-compatible chat/completions.
                                           Honors OPENAI_BASE_URL, so this also
                                           covers Azure OpenAI, Ollama, vLLM,
                                           Together, Groq, LM Studio, and any
                                           other OpenAI-API-compatible endpoint.
    "openai:<model>@<base_url>"         -> explicit base URL inline.
    "anthropic:<model>"                 -> Anthropic direct (needs `anthropic`,
                                           ANTHROPIC_API_KEY). No AWS needed.
    "bedrock:<model>", "claude"         -> AWS Bedrock (Anthropic messages API).

Bring your own model: pass a callable `teacher=my_fn` where
`my_fn(prompt: str) -> str`. It is wrapped as a provider with the same prompts,
so a local HF pipeline or an internal gateway works with zero vendor lock-in.

Each provider exposes:
    infer_schema(use_case, samples) -> Spec | None
    synthesize_states(spec, n, targets) -> list[str]
    as_teacher() -> Callable[[state_text, Question], answer]
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from .types import (
    DEFAULT_SCORE_RANGE,
    UNPARSEABLE,
    Question,
    QuestionType,
    Spec,
    validate_answer,
)


def get_provider(name) -> Optional["_BaseProvider"]:
    """Resolve a teacher spec to a provider, or None to fall back to heuristic.

    `name` may be:
      - a callable `fn(prompt: str) -> str` -> wrapped as a provider (BYO model),
      - an OpenAI-compatible id ("gpt", "openai:<model>", "openai:<model>@<url>"),
      - an Anthropic-direct id ("anthropic:<model>") -> uses ANTHROPIC_API_KEY,
      - a Bedrock id ("bedrock:<model>", "claude") -> uses AWS creds.
    """
    if name is None:
        return None
    # Bring-your-own: any callable is a complete() implementation.
    if callable(name):
        return _CallableProvider(name)
    low = str(name).lower()
    try:
        # Anthropic direct (ANTHROPIC_API_KEY) is checked before Bedrock so that
        # "anthropic:claude-..." routes to the native API, while a bare "claude"
        # or "bedrock:..." still goes through AWS Bedrock.
        if low.startswith("anthropic"):
            return _AnthropicProvider(str(name))
        if low.startswith("bedrock") or "claude" in low:
            return _BedrockProvider(str(name))
        if low.startswith(("gpt", "openai")):
            return _OpenAIProvider(str(name))
    except Exception:
        # SDK missing, bad creds, etc. -> let the caller fall back.
        return None
    return None


class _BaseProvider:
    name = "llm"

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        raise NotImplementedError

    # -- schema inference ----------------------------------------------------

    def infer_schema(self, use_case: str, samples: List[str]) -> Optional[Spec]:
        sample_block = "\n".join(f"- {s}" for s in samples[:10])
        prompt = (
            "Extract the decision schema that a use-case sentence literally "
            "states. Do NOT invent extra questions or decompose a decision into "
            "yes/no sub-questions. Return ONE question per decision the sentence "
            "asks for.\n\n"
            "Rules:\n"
            "- A list of alternatives ('route to A, B, or C') is ONE question of "
            "type \"choice\" with options [A, B, C] -- NOT several bool questions.\n"
            "- A yes/no decision ('flag X', 'decide whether Y') is type \"bool\" "
            "(no options).\n"
            "- A rating/scale ('rate 1-5', 'score urgency') is type \"score\" "
            "with score_range [lo, hi].\n"
            "- Multiple decisions joined by 'and' become multiple questions.\n"
            "- name: a short snake_case id for the decision.\n\n"
            "Examples:\n"
            "Use case: Route tickets to billing, sales, or support\n"
            "{\"questions\":[{\"name\":\"route\",\"type\":\"choice\","
            "\"options\":[\"billing\",\"sales\",\"support\"]}]}\n"
            "Use case: Flag spam and rate urgency 1-5\n"
            "{\"questions\":[{\"name\":\"flag_spam\",\"type\":\"bool\"},"
            "{\"name\":\"rate_urgency\",\"type\":\"score\",\"score_range\":[1,5]}]}\n\n"
            f"Use case: {use_case}\n"
            f"Sample inputs (for option wording only):\n{sample_block}\n\n"
            "Output ONLY the JSON, no prose, no markdown fences."
        )
        try:
            raw = self._complete(prompt)
            data = _extract_json(raw)
            questions = [_question_from_dict(q) for q in data.get("questions", [])]
            questions = [q for q in questions if q is not None]
            if not questions:
                return None
            return Spec(use_case=use_case, questions=questions)
        except Exception:
            return None

    # -- synthesis -----------------------------------------------------------

    def synthesize_states(
        self, spec: Spec, n: int, targets: Optional[Dict[str, Dict[str, int]]]
    ) -> List[str]:
        want = (
            f"Generate {n} short, realistic, in-distribution example states for "
            f"the use case: {spec.use_case}. One per line, no numbering."
        )
        try:
            raw = self._complete(want)
            lines = [ln.strip("-* \t") for ln in raw.splitlines() if ln.strip()]
            return lines[:n]
        except Exception:
            return []

    # -- as a labeling teacher ----------------------------------------------

    def as_teacher(self) -> Callable[[str, Question], Any]:
        def _teacher(text: str, q: Question) -> Any:
            prompt = _label_prompt(text, q)
            # Let real errors (auth, rate limit, network) propagate: a teacher
            # that silently returns None would poison every label with a
            # fallback answer and hide the cause. Fail loud, fix the key.
            raw = self._complete(prompt).strip()
            return _coerce_answer(raw, q)

        _teacher.name = self.name  # type: ignore[attr-defined]
        return _teacher


class _CallableProvider(_BaseProvider):
    """Wraps any user callable `fn(prompt: str) -> str` as a teacher provider.

    This is the vendor-neutral escape hatch: a local HuggingFace pipeline, an
    internal gateway, or any custom model backend works with the same prompts.
    """

    def __init__(self, fn: Callable[[str], str], name: str = "callable"):
        self._fn = fn
        self.name = getattr(fn, "name", None) or name

    def _complete(self, prompt: str) -> str:
        return self._fn(prompt)


class _OpenAIProvider(_BaseProvider):
    """OpenAI-compatible chat/completions.

    Works against any endpoint that speaks the OpenAI API, not just OpenAI
    itself: set OPENAI_BASE_URL (or append '@<base_url>' to the id) to target
    Azure OpenAI, Ollama, vLLM, Together, Groq, LM Studio, etc. The default
    model is a placeholder only used when none is given in the id; always pass
    an explicit model for non-OpenAI endpoints.
    """

    def __init__(self, name: str):
        import openai  # type: ignore  # raises if extra not installed

        # id form: "openai:<model>" or "openai:<model>@<base_url>" or "gpt"
        model_part = name.split(":", 1)[1] if ":" in name else ""
        base_url = os.getenv("OPENAI_BASE_URL")
        if "@" in model_part:
            model_part, base_url = model_part.split("@", 1)
        self._model = model_part or "gpt-4o-mini"

        # A custom base URL implies a non-OpenAI endpoint, which may not need a
        # real OpenAI key (e.g. a local Ollama server). Only require the key for
        # the default OpenAI endpoint.
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url and not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif base_url:
            kwargs["api_key"] = "not-needed"  # many local servers ignore this

        self.name = name
        self._client = openai.OpenAI(**kwargs)

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


class _BedrockProvider(_BaseProvider):
    def __init__(self, name: str):
        import boto3  # type: ignore  # raises if extra not installed

        self.name = name
        self._model = name.split(":", 1)[1] if ":" in name else "anthropic.claude-3-haiku-20240307-v1:0"
        self._client = boto3.client("bedrock-runtime")

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        resp = self._client.invoke_model(modelId=self._model, body=body)
        payload = json.loads(resp["body"].read())
        return payload.get("content", [{}])[0].get("text", "")


class _AnthropicProvider(_BaseProvider):
    """Anthropic's native Messages API (needs `anthropic`, ANTHROPIC_API_KEY).

    Distinct from `_BedrockProvider`: this talks to Anthropic directly with an
    ANTHROPIC_API_KEY, so a user can run OpenAI + Anthropic consensus with just
    the two vendor keys and no AWS setup.
    """

    def __init__(self, name: str):
        import anthropic  # type: ignore  # raises if extra not installed

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.name = name
        # id form: "anthropic:<model>"; default to a current small model.
        self._model = name.split(":", 1)[1] if ":" in name else "claude-3-5-haiku-latest"
        self._client = anthropic.Anthropic()

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        import anthropic  # type: ignore

        kwargs = dict(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        # Prefer temperature=0 for deterministic labels. Newer Claude models
        # reject the parameter outright ("temperature is deprecated for this
        # model"); on that specific 400 we retry without it. A short, tightly
        # constrained answer is effectively deterministic without it.
        try:
            resp = self._client.messages.create(temperature=0, **kwargs)
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            resp = self._client.messages.create(**kwargs)
        # content is a list of blocks; concatenate any text blocks.
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


# ---- helpers ---------------------------------------------------------------


def _label_prompt(text: str, q: Question) -> str:
    if q.type is QuestionType.CHOICE:
        opts = ", ".join(q.options)
        return (
            f"State: {text}\nQuestion: {q.prompt or q.name}\n"
            f"Answer with exactly one of: {opts}. Reply with only the option."
        )
    if q.type is QuestionType.BOOL:
        return (
            f"State: {text}\nQuestion: {q.prompt or q.name}\n"
            "Answer with only 'true' or 'false'."
        )
    lo, hi = q.score_range or DEFAULT_SCORE_RANGE
    return (
        f"State: {text}\nQuestion: {q.prompt or q.name}\n"
        f"Answer with a single number between {lo} and {hi}."
    )


def _coerce_answer(raw: str, q: Question) -> Any:
    """Normalize a teacher's raw reply, then validate it strictly.

    Providers are prompted to answer in a canonical one-token form. We validate
    the ENTIRE reply, not just its first line: a response with more than one
    non-empty line (e.g. "true\\nActually, false.") is contradictory/overlong
    and returns UNPARSEABLE rather than silently taking the first token. The
    single surviving line is stripped of wrapping punctuation and handed to the
    strict `validate_answer` boundary -- NO substring search.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) != 1:
        # Zero lines (empty) or multiple lines (extra/contradictory content).
        return UNPARSEABLE
    line = lines[0].strip()
    # Validate the raw trimmed line FIRST: an option (or bool token) may itself
    # end in punctuation, e.g. an option literally named "Yes!". Only if that
    # fails do we strip wrapping punctuation and try once more -- so punctuation
    # cleanup can rescue a noisy reply but never rejects a legitimate answer.
    answer = validate_answer(line, q)
    if answer is not UNPARSEABLE:
        return answer
    cleaned = line.strip(".!?,;:\"'()[] ").strip()
    if cleaned == line:
        return UNPARSEABLE  # nothing was stripped; no second chance needed
    return validate_answer(cleaned, q)


def _extract_json(raw: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return json.loads(m.group()) if m else {}


def _question_from_dict(d: Dict[str, Any]) -> Optional[Question]:
    try:
        qtype = QuestionType(str(d["type"]).lower())
    except (KeyError, ValueError):
        return None
    return Question(
        name=d.get("name", "q"),
        type=qtype,
        options=list(d.get("options", []) or []),
        score_range=d.get("score_range"),
        prompt=d.get("prompt", ""),
    )
