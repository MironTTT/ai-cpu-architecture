"""Затвердевание обученного графа и экспорт в формат C++-движка.

Здесь нет PyTorch: на вход приходит уже дискретная структура (списки индексов),
на выходе — либо JSON для C++, либо целочисленная симуляция.

Формат JSON (``format: "diffgraph"``)
-------------------------------------
::

    {
      "format": "diffgraph", "version": 1,
      "task": "parity4",
      "n_inputs": 4, "n_outputs": 1,
      "nodes": [
        {"op": "INPUT"}, {"op": "INPUT"}, {"op": "INPUT"}, {"op": "INPUT"},
        {"op": "XOR", "in": [0, 1]},
        ...
      ],
      "outputs": [6],
      "meta": {...}
    }

Узлы идут в топологическом порядке, поэтому ``in`` — это просто индексы более
ранних узлов, ровно как ``Node::in[]`` в C++.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .model import OPS

FORMAT = "diffgraph"
VERSION = 1


def apply_int(op: str, a: int, b: int) -> int:
    """Целочисленная семантика операции — точная копия switch в executeInto().

    Любое расхождение здесь означает, что обученный граф в C++ будет вести
    себя иначе, поэтому эта функция используется и для проверки при экспорте.
    """
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "NAND":
        return 1 - (a & b)
    if op == "ADD":
        return a + b
    if op == "MUL":
        return a * b
    if op == "COPY":
        return a
    raise ValueError(f"неизвестная операция: {op!r}")


@dataclass
class HardGraph:
    """Дискретный граф: то, что реально исполнит C++.

    Индексация: ``0 .. n_inputs-1`` — входы задачи, ``n_inputs + j`` — узел ``j``.
    """

    task: str
    n_inputs: int
    n_outputs: int
    ops: list[str] = field(default_factory=list)
    src: list[tuple[int, int]] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- построение ----------
    @classmethod
    def from_structure(
        cls,
        task: str,
        n_inputs: int,
        n_outputs: int,
        src: Sequence[Sequence[int]],
        ops: Sequence[int],
        outputs: Sequence[int],
        meta: dict[str, Any] | None = None,
    ) -> "HardGraph":
        return cls(
            task=task,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            ops=[OPS[int(i)] for i in ops],
            src=[(int(a), int(b)) for a, b in src],
            outputs=[int(o) for o in outputs],
            meta=dict(meta or {}),
        )

    @property
    def n_nodes(self) -> int:
        return len(self.ops)

    # ---------- анализ ----------
    def is_valid_dag(self) -> bool:
        for j, (a, b) in enumerate(self.src):
            node = self.n_inputs + j
            if not (0 <= a < node and 0 <= b < node):
                return False
        for o in self.outputs:
            if not (0 <= o < self.n_inputs + self.n_nodes):
                return False
        return True

    def edge_count(self) -> int:
        return 2 * self.n_nodes

    def simulate(self, inputs: Sequence[int]) -> list[int]:
        """Прогон на целых числах — то же, что сделает C++."""
        vals = [int(v) for v in inputs]
        for j, op in enumerate(self.ops):
            a, b = self.src[j]
            vals.append(apply_int(op, vals[a], vals[b]))
        return [vals[o] for o in self.outputs]

    def pruned(self) -> "HardGraph":
        """Отрезать узлы, недостижимые из выходов.

        Обучение часто оставляет «мусорные» узлы, которые никто не читает.
        Они стоят такта на каждом примере, поэтому перед экспортом убираются.
        Индексы входов при этом не меняются.
        """
        reach: set[int] = set(self.outputs)
        for j in range(self.n_nodes - 1, -1, -1):
            node = self.n_inputs + j
            if node in reach:
                a, b = self.src[j]
                reach.add(a)
                reach.add(b)

        kept = [j for j in range(self.n_nodes) if self.n_inputs + j in reach]
        remap: dict[int, int] = {i: i for i in range(self.n_inputs)}
        for new_j, j in enumerate(kept):
            remap[self.n_inputs + j] = self.n_inputs + new_j

        return HardGraph(
            task=self.task,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            ops=[self.ops[j] for j in kept],
            src=[(remap[self.src[j][0]], remap[self.src[j][1]]) for j in kept],
            outputs=[remap[o] for o in self.outputs],
            meta=dict(self.meta),
        )

    def describe(self) -> str:
        def name(idx: int) -> str:
            return f"in{idx}" if idx < self.n_inputs else f"n{idx}"

        lines = [
            f"n{self.n_inputs + j} = {op}({name(a)}, {name(b)})"
            for j, (op, (a, b)) in enumerate(zip(self.ops, self.src))
        ]
        lines += [f"out{o} = {name(i)}" for o, i in enumerate(self.outputs)]
        return "\n".join(lines)

    # ---------- сериализация ----------
    def to_dict(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = [{"op": "INPUT"} for _ in range(self.n_inputs)]
        nodes += [{"op": op, "in": [a, b]} for op, (a, b) in zip(self.ops, self.src)]
        return {
            "format": FORMAT,
            "version": VERSION,
            "task": self.task,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "nodes": nodes,
            "outputs": list(self.outputs),
            "meta": dict(self.meta),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HardGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"это не {FORMAT}-файл: {path}")

        ops: list[str] = []
        src: list[tuple[int, int]] = []
        for node in data["nodes"]:
            op = node["op"]
            if op == "INPUT":
                continue
            ops.append(op)
            ins = node.get("in", [])
            src.append((int(ins[0]), int(ins[1])))

        return cls(
            task=data["task"],
            n_inputs=int(data["n_inputs"]),
            n_outputs=int(data["n_outputs"]),
            ops=ops,
            src=src,
            outputs=[int(o) for o in data["outputs"]],
            meta=dict(data.get("meta", {})),
        )
