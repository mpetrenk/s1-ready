"""System 1 Ready (`s1-ready`).

Turn any data with a decision attached into a decontaminated, calibrated,
System 1 (decision-model)-ready dataset.

Public API:

    SchemaInferencer - infer a decision schema from a use-case prompt
    DataForge        - ingest a corpus, synthesize fillers, balance option skew
    Labeler          - teacher-consensus labeling + agreement score + review queue
    Dataset          - labeled rows + quality gates (decontaminate/balance/filter)
"""

from .types import QuestionType, Question, Spec, Row
from .schema import SchemaInferencer
from .forge import DataForge
from .labeler import Labeler
from .gates import Dataset

__version__ = "0.1.0"

__all__ = [
    "QuestionType",
    "Question",
    "Spec",
    "Row",
    "SchemaInferencer",
    "DataForge",
    "Labeler",
    "Dataset",
    "__version__",
]
