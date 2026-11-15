"""Задачи, на которых проверяется обучение топологии.

Обе задачи — булевы функции от 4 бит, и обе выбраны потому, что «алгоритм» в
них не сводится к одному узлу:

* ``parity4`` — чётность: у функции нет локального признака, XOR должен быть
  собран в цепочку/дерево.
* ``add2``    — сложение двух 2-битных чисел: требует переноса, то есть
  AND-условий между разрядами.

Полная таблица истинности — это 16 строк, поэтому её можно позволить себе
считать целиком и использовать как train и как test.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    """Описание задачи: сколько входов/выходов и как считается ответ."""

    name: str
    n_inputs: int
    n_outputs: int
    description: str

    def evaluate(self, bits: tuple[int, ...]) -> list[int]:
        raise NotImplementedError

    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """Полная таблица истинности: X [2^n, n_in], Y [2^n, n_out] (int32)."""
        xs: list[tuple[int, ...]] = []
        ys: list[list[int]] = []
        for bits in itertools.product((0, 1), repeat=self.n_inputs):
            xs.append(bits)
            ys.append(self.evaluate(bits))
        return (
            np.asarray(xs, dtype=np.int32),
            np.asarray(ys, dtype=np.int32),
        )

    @property
    def n_examples(self) -> int:
        return 1 << self.n_inputs


@dataclass(frozen=True)
class Parity(Task):
    """Чётность n бит: 1, если единиц нечётное число."""

    n_bits: int = 4

    def __init__(self, n_bits: int = 4) -> None:
        object.__setattr__(self, "name", f"parity{n_bits}")
        object.__setattr__(self, "n_inputs", n_bits)
        object.__setattr__(self, "n_outputs", 1)
        object.__setattr__(self, "description", f"чётность {n_bits} бит (XOR всех входов)")
        object.__setattr__(self, "n_bits", n_bits)

    def evaluate(self, bits: tuple[int, ...]) -> list[int]:
        acc = 0
        for b in bits:
            acc ^= b
        return [acc]


@dataclass(frozen=True)
class Add2(Task):
    """Сложение двух 2-битных чисел.

    Входы:  a0 a1 b0 b1  (младшие разряды первыми)
    Выходы: s0 s1 s2     (3 бита суммы, 0..6)
    """

    def __init__(self) -> None:
        object.__setattr__(self, "name", "add2")
        object.__setattr__(self, "n_inputs", 4)
        object.__setattr__(self, "n_outputs", 3)
        object.__setattr__(self, "description", "сложение двух 2-битных чисел (a0 a1 b0 b1 -> s0 s1 s2)")

    def evaluate(self, bits: tuple[int, ...]) -> list[int]:
        a = bits[0] | (bits[1] << 1)
        b = bits[2] | (bits[3] << 1)
        s = a + b
        return [(s >> k) & 1 for k in range(3)]


TASKS: dict[str, Task] = {
    "parity4": Parity(4),
    "add2": Add2(),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise SystemExit(
            f"Неизвестная задача: {name!r}. Доступны: {', '.join(sorted(TASKS))}"
        ) from None
