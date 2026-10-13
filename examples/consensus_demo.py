"""Two-teacher consensus labeling demo (OpenAI + Anthropic).

Run from the repo root with both keys exported:

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...   # must be WORKSPACE-scoped
    PYTHONPATH=. python3 examples/consensus_demo.py

Labels 5 SMS messages with two independent LLM teachers, then prints each
model's vote, the consensus gold answer, and the agreement score per row.
With two teachers a BOOL agreement is 1.0 (both agree) or 0.5 (they split);
any split lands in the review queue.
"""

import os

from s1_ready.forge import DataForge
from s1_ready.labeler import Labeler
from s1_ready.types import Question, QuestionType, Spec

# Override either model without editing this file:
#   OPENAI_MODEL=gpt-4o-mini  ANTHROPIC_MODEL=claude-3-5-haiku-20241022
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

spec = Spec(
    use_case="Flag whether an SMS message is spam",
    questions=[
        Question(
            name="is_spam",
            type=QuestionType.BOOL,
            prompt="Is this SMS message spam?",
        )
    ],
)

msgs = [
    "WINNER!! You have been selected to receive a 900 prize. Call now!",
    "Hey, are we still on for lunch tomorrow at noon?",
    "URGENT: your account has been suspended, verify at http://bit.ly/x",
    "Can you send me the slides from the meeting?",
    "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
]

forge = DataForge(spec).ingest(msgs)
lab = Labeler(
    teacher=[f"openai:{OPENAI_MODEL}", f"anthropic:{ANTHROPIC_MODEL}"]
)
labeled = lab.label(forge)

for row in labeled.rows:
    print(repr(row.state[:55]))
    print("  gold:     ", row.gold)
    print("  votes:    ", row.votes)
    print("  agreement:", row.agreement)
    print()

print("agreement summary:", labeled.agreement_summary())
print("review queue (<0.7):", len(labeled.review_queue()))
