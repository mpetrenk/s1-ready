"""Format adapters: one Dataset -> many training formats.

A single labeled Dataset can be emitted in several shapes depending on the
trainer downstream. All formats are newline-delimited JSON (JSONL).

Formats:

  "typed-decisions"  (default) -- the native s1-ready shape. One object per row:
        {"state": str,
         "questions": [ {name, type, options?/score_range?}, ... ],
         "gold": {qname: answer, ...},
         "agreement": {qname: float, ...},
         "synthetic": bool}
     Carries the full schema + confidence so a Strands Decider-style model has
     everything it needs, including per-row agreement for curriculum/weighting.

  "unsloth"          -- Alpaca-style instruction tuning (widely read: Unsloth,
     TRL, Axolotl):
        {"instruction": <rendered question(s)>,
         "input": <state>,
         "output": <rendered gold answer(s)>}

  "conversational"   -- ShareGPT-style chat transcript, the format Unsloth now
     recommends by default (also read by TRL / Axolotl):
        {"conversations": [
            {"role": "user", "content": <question(s) + state>},
            {"role": "assistant", "content": <rendered gold answer(s)>}]}

  "raw"              -- minimal {state, gold} pairs, nothing else.

Public functions:
    to_records(dataset, fmt)        -> list[dict]
    write_jsonl(dataset, path, fmt, emit_helper=False)
        -> path  (also writes a sidecar <path>.spec.json; optionally a train
                  helper script for the Unsloth/TRL handoff)
    write_helper_script(dataset, data_path, fmt) -> helper_path
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

_FORMATS = ("typed-decisions", "unsloth", "conversational", "raw")

# Formats for which a train-helper script (Option C) can be generated.
_HELPER_FORMATS = ("unsloth", "conversational")


def to_records(dataset, fmt: str = "typed-decisions") -> List[Dict[str, Any]]:
    if fmt not in _FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {_FORMATS}")
    if fmt == "typed-decisions":
        return [_typed_record(dataset, r) for r in dataset.rows]
    if fmt == "unsloth":
        return [_unsloth_record(dataset, r) for r in dataset.rows]
    if fmt == "conversational":
        return [_conversational_record(dataset, r) for r in dataset.rows]
    return [{"state": r.state, "gold": r.gold} for r in dataset.rows]


def write_jsonl(
    dataset, path: str, fmt: str = "typed-decisions", emit_helper: bool = False
) -> str:
    records = to_records(dataset, fmt=fmt)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Sidecar: persist the schema next to the data so the format is self-describing.
    spec_path = f"{path}.spec.json"
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(dataset.spec.to_dict(), fh, ensure_ascii=False, indent=2)

    # Option C: optional train-helper script for the training-framework handoff.
    if emit_helper:
        if fmt in _HELPER_FORMATS:
            write_helper_script(dataset, path, fmt)
        # For non-helper formats we silently skip; the caller decides whether to warn.

    return path


# ---- per-format record builders --------------------------------------------


def _typed_record(dataset, row) -> Dict[str, Any]:
    return {
        "state": row.state,
        "questions": [q.to_dict() for q in dataset.spec.questions],
        "gold": _jsonable(row.gold),
        "agreement": _jsonable(row.agreement),
        "synthetic": row.synthetic,
    }


def _unsloth_record(dataset, row) -> Dict[str, Any]:
    instruction, output = _render_turns(dataset, row)
    return {
        "instruction": instruction,
        "input": row.state,
        "output": output,
    }


def _conversational_record(dataset, row) -> Dict[str, Any]:
    instruction, output = _render_turns(dataset, row)
    # The user turn carries the task (questions) plus the state to judge.
    user_content = f"{instruction}\n\n{row.state}".strip()
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def _render_turns(dataset, row):
    """Shared rendering: (instruction block, gold-answer block) for a row.

    Reused by the Alpaca (`unsloth`) and ShareGPT (`conversational`) formats so
    the two stay consistent.
    """
    instructions = []
    outputs = []
    for q in dataset.spec.questions:
        instructions.append(_render_question(q))
        outputs.append(f"{q.name}: {_render_answer(row.gold.get(q.name))}")
    return "\n".join(instructions), "\n".join(outputs)


def _render_question(q) -> str:
    base = q.prompt or q.name
    from .types import QuestionType

    if q.type is QuestionType.CHOICE and q.options:
        return f"{base} (choose one: {', '.join(q.options)})"
    if q.type is QuestionType.BOOL:
        return f"{base} (true or false)"
    if q.type is QuestionType.SCORE:
        lo, hi = q.score_range or [1.0, 5.0]
        return f"{base} (score {lo}-{hi})"
    return base


def _render_answer(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[k] = v
    return out


# ---- Option C: train-helper script generation ------------------------------


def write_helper_script(dataset, data_path: str, fmt: str) -> str:
    """Write a ready-to-run `<data>.train.py` next to the exported JSONL.

    The script loads the JSONL, applies the model's chat template via a
    `formatting_prompts_func`, and hands the result to TRL's `SFTTrainer`
    (the same trainer Unsloth patches). It is boilerplate the user would
    otherwise write by hand; everything model-specific is a clearly marked
    TODO at the top (base model id).

    Returns the helper path. Raises ValueError for formats without a helper.
    """
    if fmt not in _HELPER_FORMATS:
        raise ValueError(
            f"no train helper for format {fmt!r}; expected one of {_HELPER_FORMATS}"
        )
    helper_path = f"{data_path}.train.py"
    data_basename = os.path.basename(data_path)
    body = _ALPACA_HELPER if fmt == "unsloth" else _CONVERSATIONAL_HELPER
    script = body.format(data_file=data_basename, use_case=dataset.spec.use_case)
    with open(helper_path, "w", encoding="utf-8") as fh:
        fh.write(script)
    return helper_path


# NOTE: these templates are emitted verbatim as standalone scripts. They use
# `{{` / `}}` to escape literal braces so str.format only fills {data_file} and
# {use_case}. They import unsloth/trl, which the END USER installs in their
# training environment -- they are NOT dependencies of s1-ready itself.

_ALPACA_HELPER = '''"""Auto-generated by s1-ready. Fine-tune on an Alpaca-style ({{instruction, input, output}}) export.

Use case: {use_case}

Run this in your TRAINING environment (where unsloth + trl + a GPU live):
    pip install unsloth trl datasets
    python {data_file}.train.py

Edit MODEL_NAME below before running.
"""

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# TODO: set your base model.
MODEL_NAME = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
DATA_FILE = "{data_file}"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")


def formatting_prompts_func(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {{"role": "user", "content": f"{{instruction}}\\n\\n{{inp}}"}},
            {{"role": "assistant", "content": output}},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {{"text": texts}}


dataset = dataset.map(formatting_prompts_func, batched=True)

# Sanity-check one rendered row before training.
print("=== sample rendered row ===")
print(dataset[0]["text"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()
'''

_CONVERSATIONAL_HELPER = '''"""Auto-generated by s1-ready. Fine-tune on a ShareGPT-style ({{conversations: [...]}}) export.

Use case: {use_case}

Run this in your TRAINING environment (where unsloth + trl + a GPU live):
    pip install unsloth trl datasets
    python {data_file}.train.py

Edit MODEL_NAME below before running.
"""

from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig

# TODO: set your base model.
MODEL_NAME = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
DATA_FILE = "{data_file}"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
# Normalize the ShareGPT rows, then render with the model's chat template.
dataset = standardize_sharegpt(dataset)


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in examples["conversations"]
    ]
    return {{"text": texts}}


dataset = dataset.map(formatting_prompts_func, batched=True)

# Sanity-check one rendered row before training.
print("=== sample rendered row ===")
print(dataset[0]["text"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()
'''
