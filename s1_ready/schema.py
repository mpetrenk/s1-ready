"""Schema inference: a use-case sentence -> a decision `Spec`.

This is the step people get wrong by hand. A sentence like

    "Route support tickets to a team and flag refunds"

implies two questions:

    - a CHOICE  ("route ... to a team")  with an option set of team names
    - a BOOL    ("flag refunds")

`SchemaInferencer.from_prompt` turns the sentence into a `Spec`. It ships with a
zero-dependency heuristic that:

  1. splits the use case into clauses (on "and", ",", "then", ";"),
  2. classifies each clause into CHOICE / BOOL / SCORE from verb/keyword cues,
  3. proposes an option set for CHOICE clauses, mining candidate options from a
     sample of the real corpus when one is provided (so options are
     in-distribution, not guessed), and falling back to generic placeholders.

An optional LLM teacher can replace the heuristic for richer inference; pass
`teacher="gpt"` (requires the `[llm]` extra). The heuristic keeps the vertical
slice runnable with no API keys.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Union

from .types import Question, QuestionType, Spec

# ---- keyword cues for question-type classification -------------------------

# Cue words/phrases per type. Matched on WORD BOUNDARIES (see _has_cue) so that
# e.g. "rate" does not match inside "AI-generated".
_BOOL_CUES = (
    "flag", "detect", "whether", "should", "yes/no", "yes or no",
    "mark", "escalate", "decide if", "decide whether", "determine if",
    "determine whether", "is it", "does it", "contains", "qualifies",
    "requires", "needs", "valid", "duplicate", "overdue", "suspicious",
    "fraudulent", "toxic", "fake", "strong", "covered", "approve",
)
_SCORE_CUES = (
    "score", "rate", "rating", "priority", "severity", "urgency",
    "confidence", "risk level", "how likely", "scale of", "scale from",
    "0-", "1-5", "1-10", "1 to", "0 to", "stars", "from 1", "out of",
)
# CHOICE verbs are the strongest signal: an explicit classification verb outranks
# a stray SCORE/BOOL keyword that may also appear in the clause.
_CHOICE_CUES = (
    "route", "classify", "categorize", "assign", "pick", "choose", "select",
    "which", "label as", "sort into", "sort the", "bucket", "categorize",
    "sentiment as",
)

# Verbs that begin a new decision clause. Used to tell a clause-separating comma
# ("rate X, flag Y") from an option-separating comma ("billing, sales, support").
_DECISION_VERBS = (
    "rate", "score", "classify", "categorize", "route", "assign", "flag",
    "detect", "decide", "determine", "identify", "mark", "tag", "sort",
    "pick", "choose", "select", "label", "assess", "check", "estimate",
    "predict", "judge",
)
_DECISION_VERBS_RE = "|".join(_DECISION_VERBS)

# Clauses are separate decisions. We split only on true clause boundaries
# (";", "then", and a top-level "and"). We deliberately do NOT split on "," or
# "or": those usually separate *options within one clause* ("route to billing,
# sales, or support"), and splitting them would shatter one CHOICE question into
# several bogus ones. An "and" that is part of an inline option list is rejoined
# in _split_clauses below.
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|\bthen\b|\band\b)\s*", flags=re.IGNORECASE)

# A clause introduces an inline option list when it contains a listing verb
# followed by "to/into/as/among/between".
_LIST_INTRO = re.compile(
    r"\b(?:to|into|as|among|between)\b", flags=re.IGNORECASE
)

# Stopwords for mining option candidates from the corpus / prompt. These are
# words that make poor CHOICE *option* values (generic verbs, articles, the
# domain noun "ticket"), so they are filtered out of mined/explicit options.
_STOP = {
    "the", "a", "an", "to", "of", "for", "and", "or", "is", "are", "be", "in",
    "on", "at", "it", "this", "that", "with", "as", "by", "from", "team",
    "flag", "route", "ticket", "tickets", "please", "need", "want", "should",
    "can", "was", "were", "them", "their", "my", "your", "our",
}

# A smaller set used only when deriving a question *name* from the clause. We
# keep content words like "route"/"flag" here (they describe the decision) and
# only drop function words and the filler noun "ticket(s)".
_NAME_STOP = {
    "the", "a", "an", "to", "of", "for", "and", "or", "is", "are", "be", "in",
    "on", "at", "it", "this", "that", "with", "as", "by", "from",
    "ticket", "tickets", "please",
}

# Leading words that can cling to the FIRST option in an inline list and should
# be stripped (verbs, articles, superlatives). "not" is intentionally absent so
# "not spam" survives as a two-word option.
_OPT_INTRO_WORDS = {
    "the", "a", "an", "best", "category", "categories", "it", "its", "class",
    "type", "label", "bucket", "into", "as", "to", "of",
    "classify", "categorize", "sort", "route", "assign", "pick", "choose",
    "select", "tag", "mark", "label",
}

# Pure connective/function tokens that cannot stand alone as an option. A
# candidate made ENTIRELY of these is rejected (a dangling "or"), but a mix like
# "not spam" is kept.
_OPT_STOP = {
    "or", "and", "the", "a", "an", "to", "of", "for", "as", "into", "with",
}


class SchemaInferencer:
    """Infer a decision `Spec` from a natural-language use case."""

    @classmethod
    def from_prompt(
        cls,
        use_case: str,
        sample_states: Optional[Union[Iterable[str], "object"]] = None,
        teacher: Optional[Union[str, Callable]] = None,
        max_options: int = 6,
    ) -> Spec:
        """Return a `Spec` for `use_case`.

        sample_states : an iterable of state strings, or a file-like/open()
                        handle; a few are read to propose in-distribution
                        CHOICE options.
        teacher       : optional LLM id (e.g. "gpt", "bedrock:claude"). When set
                        and the `[llm]` extra is installed, defers to the LLM.
                        Falls back to the heuristic otherwise.
        """
        samples = cls._read_samples(sample_states)

        if teacher:
            spec = cls._infer_with_llm(use_case, samples, teacher)
            if spec is not None:
                return spec
            # fall through to heuristic if the teacher is unavailable

        questions = cls._infer_heuristic(use_case, samples, max_options)
        return Spec(use_case=use_case.strip(), questions=questions)

    # -- heuristic path ------------------------------------------------------

    @classmethod
    def _infer_heuristic(
        cls, use_case: str, samples: Sequence[str], max_options: int
    ) -> List[Question]:
        clauses = cls._split_clauses(use_case)
        if not clauses:
            clauses = [use_case.strip()]

        questions: List[Question] = []
        for i, clause in enumerate(clauses):
            clause = clause.strip(" ,;.")  # trim connective punctuation
            qtype = cls._classify_clause(clause)
            name = cls._clause_to_name(clause, fallback=f"q{i + 1}")

            if qtype is QuestionType.BOOL:
                questions.append(
                    Question(name=name, type=qtype, prompt=clause)
                )
            elif qtype is QuestionType.SCORE:
                questions.append(
                    Question(
                        name=name,
                        type=qtype,
                        score_range=[1.0, 5.0],
                        prompt=clause,
                    )
                )
            else:  # CHOICE
                options = cls._propose_options(clause, samples, max_options)
                questions.append(
                    Question(
                        name=name,
                        type=qtype,
                        options=options,
                        prompt=clause,
                    )
                )
        return questions

    @classmethod
    def _split_clauses(cls, use_case: str) -> List[str]:
        """Split a use case into clauses, each a distinct decision.

        Splits on ";", "then", and "and", then rejoins any fragment that is
        really the tail of an inline option list belonging to the previous
        clause. A fragment is treated as an option tail (not a new clause) when
        the previous clause opened an option list (contains to/into/as/...) and
        the fragment is short and has no decision verb of its own.
        """
        # First, promote commas that separate CLAUSES (not options) to clause
        # boundaries. A comma starts a new clause when the text after it begins
        # with a decision verb ("..., flag if ...", "..., and classify ...").
        # A comma inside an option list ("billing, sales, or support") does not,
        # because "sales" is not a decision verb. We insert a ';' there so the
        # existing splitter picks it up.
        prepared = re.sub(
            r",\s*(?=(?:and\s+|then\s+)?(?:" + _DECISION_VERBS_RE + r")\b)",
            "; ",
            use_case,
            flags=re.IGNORECASE,
        )

        raw = [c.strip() for c in _CLAUSE_SPLIT.split(prepared) if c.strip()]
        if len(raw) <= 1:
            return raw

        merged: List[str] = []
        for frag in raw:
            if merged and cls._is_option_tail(merged[-1], frag):
                # Reattach to the previous clause as part of its option list.
                merged[-1] = f"{merged[-1]}, {frag}"
            else:
                merged.append(frag)
        return merged

    @staticmethod
    def _is_option_tail(prev: str, frag: str) -> bool:
        prev_low = prev.lower()
        frag_low = frag.lower()
        opened_list = bool(_LIST_INTRO.search(prev_low))
        has_own_verb = any(
            c in frag_low for c in (_CHOICE_CUES + _BOOL_CUES + _SCORE_CUES)
        )
        # A trailing option is a short noun phrase (<=3 words) with no verb and
        # follows a clause that opened an option list.
        short = len(re.findall(r"[a-zA-Z]+", frag_low)) <= 3
        return opened_list and short and not has_own_verb

    @staticmethod
    def _has_cue(low: str, cues) -> bool:
        """True if any cue appears as a whole word / phrase in `low`.

        Word-boundary matched so "rate" does not fire inside "AI-generated" and
        "is" does not fire inside "this".
        """
        for cue in cues:
            if re.search(r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])", low):
                return True
        return False

    @classmethod
    def _classify_clause(cls, clause: str) -> QuestionType:
        low = clause.lower()
        has_options = len(cls._explicit_options(clause)) >= 2

        # Precedence, strongest evidence first:
        # 1. An explicit inline option list -> CHOICE, regardless of stray cues
        #    ("... positive, negative, or neutral" even though "sentiment" is a
        #    SCORE cue).
        if has_options:
            return QuestionType.CHOICE
        # 2. An explicit CHOICE verb ("classify", "categorize", ...) outranks a
        #    lone SCORE/BOOL keyword ("classify the sentiment" -> CHOICE).
        if cls._has_cue(low, _CHOICE_CUES):
            return QuestionType.CHOICE
        # 3. SCORE cues (numeric ranges, "rate", "scale of", ...).
        if cls._has_cue(low, _SCORE_CUES):
            return QuestionType.SCORE
        # 4. BOOL cues (yes/no framings).
        if cls._has_cue(low, _BOOL_CUES):
            return QuestionType.BOOL
        # 5. Default: a classification decision is the common case.
        return QuestionType.CHOICE

    @staticmethod
    def _clause_to_name(clause: str, fallback: str) -> str:
        # Name the question after the decision itself (the verb phrase before any
        # option list), not the option values. Drop the inline list tail so
        # "route support tickets to billing, sales, ..." names 'route_support',
        # not 'billing'.
        head = _LIST_INTRO.split(clause, maxsplit=1)[0]
        words = [w for w in re.findall(r"[a-zA-Z]+", head.lower()) if w not in _NAME_STOP]
        if not words:
            # Fall back to the whole clause (minus stopwords) for BOOL/SCORE.
            words = [w for w in re.findall(r"[a-zA-Z]+", clause.lower()) if w not in _NAME_STOP]
        if not words:
            return fallback
        name = "_".join(words[:3])
        return name or fallback

    @classmethod
    def _propose_options(
        cls, clause: str, samples: Sequence[str], max_options: int
    ) -> List[str]:
        # 1) explicit list in the clause: "route to billing, support, or sales"
        explicit = cls._explicit_options(clause)
        if explicit:
            return explicit[:max_options]

        # 2) mine frequent salient tokens from the corpus sample
        mined = cls._mine_options(samples, max_options)
        if mined:
            return mined

        # 3) generic placeholders (clearly flagged for the user to edit)
        return [f"option_{i + 1}" for i in range(min(3, max_options))]

    @staticmethod
    def _explicit_options(clause: str) -> List[str]:
        """Extract an inline option list, e.g. '... as X, Y, or Z'.

        Handles multi-word options ("needs review", "car rental"), alphanumeric
        labels ("P0", "tier 1"), and the first option even when it is glued to
        the list-introducing verb ("category: electronics, ..." / "into bug,
        ..."). Returns [] unless at least two clean options are found.
        """
        # Trim everything up to and including the LAST list-intro token
        # (to / into / as / among / between) or a colon, so the options start
        # clean. Fall back to the whole clause if no intro is present.
        m = re.search(
            r"\b(?:to|into|as|among|between)\b\s*:?\s*(.+)$|:\s*(.+)$",
            clause,
            flags=re.IGNORECASE,
        )
        tail = (m.group(1) or m.group(2)) if m else clause
        tail = tail.strip(" .")

        if not m:
            # No "as/to/into" intro. Strip a leading question/verb stem like
            # "is this message", "does the text" so "Is this a question or a
            # statement" yields options from "a question or a statement".
            tail = re.sub(
                r"^(?:is|are|was|were|does|do|should|would|will|can)\b"
                r"(?:\s+(?:this|that|the|a|an|it|its|message|text|item|input))*\s+",
                "",
                tail,
                flags=re.IGNORECASE,
            )

        # Split on list separators: comma, slash, "or", "and".
        parts = re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", tail, flags=re.IGNORECASE)

        opts: List[str] = []
        for p in parts:
            opt = SchemaInferencer._clean_option(p)
            if opt:
                opts.append(opt)
        # De-dup while preserving order.
        seen = set()
        deduped = []
        for o in opts:
            if o not in seen:
                seen.add(o)
                deduped.append(o)
        return deduped if len(deduped) >= 2 else []

    # Trailing filler words that are part of the sentence, not the option
    # (e.g. "to tier 1, tier 2, or tier 3 support" -> "support" is filler).
    _OPT_TRAILING_FILLER = {"support", "team", "bucket"}

    @staticmethod
    def _clean_option(phrase: str) -> str:
        """Normalize one candidate option phrase, or '' if it isn't a real option."""
        p = phrase.strip(" .\t").lower()
        if not p:
            return ""
        # Tokens may be alphanumeric (P0, tier1) or words; keep hyphens.
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", p)
        # Drop leading intro/stopwords that can cling to the first option
        # ("the best category electronics" -> "electronics"; a bare "not" stays
        # so "not spam" survives).
        while tokens and tokens[0] in _OPT_INTRO_WORDS:
            tokens.pop(0)
        # Drop a single trailing filler noun ("tier 3 support" -> "tier 3").
        if len(tokens) > 1 and tokens[-1] in SchemaInferencer._OPT_TRAILING_FILLER:
            tokens = tokens[:-1]
        if not tokens:
            return ""
        # Reject fragments that are entirely connective/stopwords (e.g. a dangling
        # "or"), but allow meaningful multi-word options like "not spam".
        if all(t in _OPT_STOP for t in tokens):
            return ""
        return " ".join(tokens)

    @staticmethod
    def _mine_options(samples: Sequence[str], max_options: int) -> List[str]:
        if not samples:
            return []
        counter: Counter = Counter()
        for s in samples[:200]:
            for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", s.lower()):
                if tok not in _STOP:
                    counter[tok] += 1
        common = [tok for tok, n in counter.most_common(max_options * 2) if n >= 2]
        return common[:max_options]

    # -- optional LLM path ---------------------------------------------------

    @classmethod
    def _infer_with_llm(
        cls, use_case: str, samples: Sequence[str], teacher
    ) -> Optional[Spec]:
        """Defer to an LLM teacher if the optional provider is available.

        Returns None (so the caller falls back to the heuristic) when the
        provider or credentials are missing.
        """
        try:
            from .providers import get_provider  # optional, [llm] extra
        except Exception:
            return None
        provider = get_provider(teacher)
        if provider is None:
            return None
        return provider.infer_schema(use_case, list(samples))

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _read_samples(sample_states) -> List[str]:
        if sample_states is None:
            return []
        # Accept an open file handle / anything with .read()
        if hasattr(sample_states, "read"):
            text = sample_states.read()
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Accept a plain iterable of strings
        if isinstance(sample_states, str):
            return [ln.strip() for ln in sample_states.splitlines() if ln.strip()]
        return [str(s).strip() for s in sample_states if str(s).strip()]
