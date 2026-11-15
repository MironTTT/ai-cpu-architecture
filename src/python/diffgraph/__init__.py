"""Дифференцируемый граф: обучение топологии через Gumbel-Softmax.

Пакет решает одну задачу: найти структуру вычислительного графа градиентным
спуском, а затем отдать её в C++-движок как обычный жёсткий граф.

::

    python -m diffgraph --task parity4
    python -m diffgraph --all
"""

from .export import HardGraph, apply_int
from .model import OPS, DifferentiableGraph
from .refine import RefineResult, TruthTable, refine
from .tasks import TASKS, Task, get_task
from .train import Candidate, TrainConfig, TrainResult, evaluate, train

__all__ = [
    "OPS",
    "Candidate",
    "DifferentiableGraph",
    "HardGraph",
    "RefineResult",
    "TASKS",
    "Task",
    "TrainConfig",
    "TrainResult",
    "TruthTable",
    "apply_int",
    "evaluate",
    "get_task",
    "refine",
    "train",
]
