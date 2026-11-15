"""Дискретная добивка структуры: поиск по таблицам истинности.

Зачем это нужно
---------------
Градиент двигает логиты каждого узла **независимо**. Поэтому он хорошо находит
структуру «в целом», но застревает там, где нужно согласованно изменить
несколько узлов сразу. Классический пример — перенос в сложении: чтобы довести
схему до конца, надо одновременно создать узел ``AND(c1, a1^b1)`` и подключить
его к выходу ``s2``. Пока узел не подключён к выходу, его градиент равен нулю,
а подключение — отдельное дискретное действие.

Это та же проблема, что и у DARTS: после дифференцируемого поиска архитектуру
дискретизируют и доучивают. Здесь «доучивание» — локальный поиск по дискретным
правкам графа.

Почему это быстро
-----------------
Задача маленькая: 4 входа — это 16 строк таблицы истинности. Значит значение
любого узла — это 16-битная маска, а операция над узлами — одна побитовая
операция над целым числом. Проверка целой структуры стоит несколько десятков
таких операций, поэтому можно честно перебрать **всё** соседство (смена
операции, переподключение ребра, смена выхода, добавление узла, вставка узла
в выход) и брать строго лучшее.

Базис поиска — булев: AND, OR, XOR, NAND, COPY. Это не ограничение, а
требование: экспортируемый граф обязан оставаться в {0, 1}, иначе побитовые
операции C++ перестанут совпадать с полиномиальными операциями обучения.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .export import HardGraph, apply_int
from .tasks import Task

#: Операции, допустимые в добивке (все сохраняют значения в {0, 1}).
BOOLEAN_OPS: tuple[str, ...] = ("AND", "OR", "XOR", "NAND", "COPY")

#: Как заменить арифметические операции при переходе к булеву базису.
#: На значениях 0/1 это тождественная замена.
_ARITH_TO_BOOL = {"ADD": "OR", "MUL": "AND"}


@dataclass
class RefineResult:
    graph: HardGraph
    accuracy: float
    correct: int
    total: int
    steps: int
    start_accuracy: float
    boolean: bool


# ---------------------------------------------------------------------------
class TruthTable:
    """Задача, переведённая в битовые маски.

    ``in_vecs[i]`` — маска входа ``i``: бит ``r`` равен значению входа на строке
    ``r`` таблицы истинности. ``target[o]`` — такая же маска для выхода ``o``.
    """

    def __init__(self, task: Task) -> None:
        x, y = task.table()
        self.n_inputs = task.n_inputs
        self.n_outputs = task.n_outputs
        self.n_rows = len(x)
        self.full = (1 << self.n_rows) - 1

        self.in_vecs: list[int] = []
        for i in range(task.n_inputs):
            mask = 0
            for r, row in enumerate(x):
                if int(row[i]):
                    mask |= 1 << r
            self.in_vecs.append(mask)

        self.target: list[int] = []
        for o in range(task.n_outputs):
            mask = 0
            for r, row in enumerate(y):
                if int(row[o]):
                    mask |= 1 << r
            self.target.append(mask)

    def masks(self, ops: tuple[str, ...], src: tuple[tuple[int, int], ...]) -> list[int]:
        vals = list(self.in_vecs)
        for j, op in enumerate(ops):
            a, b = src[j]
            va, vb = vals[a], vals[b]
            if op == "AND":
                v = va & vb
            elif op == "OR":
                v = va | vb
            elif op == "XOR":
                v = va ^ vb
            elif op == "NAND":
                v = (~(va & vb)) & self.full
            else:  # COPY
                v = va
            vals.append(v)
        return vals

    def score(
        self,
        ops: tuple[str, ...],
        src: tuple[tuple[int, int], ...],
        outs: tuple[int, ...],
    ) -> int:
        """Сколько строк таблицы истинности решены полностью верно."""
        vals = self.masks(ops, src)
        ok = self.full
        for o, tgt in zip(outs, self.target):
            ok &= ~(vals[o] ^ tgt) & self.full
        return ok.bit_count()


# ---------------------------------------------------------------------------
def accuracy(graph: HardGraph, x: np.ndarray, y: np.ndarray) -> float:
    """Доля примеров, где верны ВСЕ выходные биты."""
    ok = 0
    for row_x, row_y in zip(x, y):
        if graph.simulate(row_x) == [int(v) for v in row_y]:
            ok += 1
    return ok / len(x)


def bit_accuracy(graph: HardGraph, x: np.ndarray, y: np.ndarray) -> float:
    """Доля верных отдельных выходных битов."""
    total = correct = 0
    for row_x, row_y in zip(x, y):
        for got, want in zip(graph.simulate(row_x), row_y):
            total += 1
            correct += int(got == int(want))
    return correct / total if total else 0.0


def boolean_domain(graph: HardGraph, x: np.ndarray) -> bool:
    """Остаются ли все значения графа в {0, 1}."""
    for row in x:
        vals = [int(v) for v in row]
        for j, op in enumerate(graph.ops):
            a, b = graph.src[j]
            vals.append(apply_int(op, vals[a], vals[b]))
        if any(v not in (0, 1) for v in vals):
            return False
    return True


# ---------------------------------------------------------------------------
def _to_boolean_basis(graph: HardGraph) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Привести структуру к булеву базису (ADD→OR, MUL→AND)."""
    ops = tuple(_ARITH_TO_BOOL.get(op, op) for op in graph.ops)
    src = tuple((int(a), int(b)) for a, b in graph.src)
    outs = tuple(int(o) for o in graph.outputs)
    return ops, src, outs


def _neighbors(
    ops: tuple[str, ...],
    src: tuple[tuple[int, int], ...],
    outs: tuple[int, ...],
    n_inputs: int,
    max_nodes: int,
):
    """Все структуры, отличающиеся одной правкой."""
    m = len(ops)
    limit = n_inputs + m

    # 1) смена операции узла
    for j in range(m):
        for op in BOOLEAN_OPS:
            if op != ops[j]:
                yield (ops[:j] + (op,) + ops[j + 1:], src, outs)

    # 2) переподключение одного входа узла (только на более ранний — DAG)
    for j in range(m):
        for slot in (0, 1):
            a, b = src[j]
            for v in range(n_inputs + j):
                if v == src[j][slot]:
                    continue
                ns = (v, b) if slot == 0 else (a, v)
                yield (ops, src[:j] + (ns,) + src[j + 1:], outs)

    # 3) смена выхода
    for o in range(len(outs)):
        for v in range(limit):
            if v != outs[o]:
                yield (ops, src, outs[:o] + (v,) + outs[o + 1:])

    # 4) добавление узла
    if m < max_nodes:
        for op in BOOLEAN_OPS:
            for a in range(limit):
                for b in range(a, limit):
                    yield (ops + (op,), src + ((a, b),), outs)

    # 5) вставка узла прямо в выход: out[o] = op1(out[o], op2(a, b))
    if m + 2 <= max_nodes:
        for o in range(len(outs)):
            old = outs[o]
            for op2 in BOOLEAN_OPS:
                for a in range(limit):
                    for b in range(a, limit):
                        for op1 in ("OR", "XOR"):
                            yield (
                                ops + (op2, op1),
                                src + ((a, b), (old, n_inputs + m)),
                                outs[:o] + (n_inputs + m + 1,) + outs[o + 1:],
                            )


def _search(
    tt: TruthTable,
    start: tuple[tuple[str, ...], tuple[tuple[int, int], ...], tuple[int, ...]],
    max_steps: int,
    limit: int,
    seed: int,
    plateau: int,
) -> tuple[tuple, int, int]:
    """Один проход локального поиска. Возвращает (структура, счёт, шагов)."""
    rng = random.Random(seed)
    ops, src, outs = start
    best = start
    best_correct = tt.score(*start)
    steps = 0
    stale = 0

    for _ in range(max_steps):
        if best_correct >= tt.n_rows:
            break
        steps += 1
        cur_score = tt.score(ops, src, outs)
        improving = None
        improving_score = cur_score
        neutral: list[tuple] = []

        for cand in _neighbors(ops, src, outs, tt.n_inputs, limit):
            s = tt.score(*cand)
            if s > improving_score:
                improving_score, improving = s, cand
            elif s == improving_score and improving is not None and len(cand[0]) < len(improving[0]):
                improving = cand            # та же точность — берём схему проще
            elif s == cur_score:
                neutral.append(cand)

        if improving is not None:
            ops, src, outs = improving
            if improving_score > best_correct:
                best_correct = improving_score
                best = improving
                stale = 0
        elif neutral:
            # Плато: предпочитаем шаги, которые наращивают схему, иначе узел,
            # который ещё ни к чему не подключён, никогда не появится.
            grown = [c for c in neutral if len(c[0]) > len(ops)]
            ops, src, outs = rng.choice(grown if grown else neutral)
            stale += 1
        else:
            break

        if stale > plateau:
            ops, src, outs = best
            stale = 0

    return best, best_correct, steps


def refine(
    graph: HardGraph,
    task: Task,
    max_steps: int = 120,
    seed: int = 0,
    max_nodes: int | None = None,
    plateau: int = 40,
    restarts: int = 6,
) -> RefineResult:
    """Локальный поиск по дискретным правкам структуры.

    На каждом шаге перебирается всё соседство; берётся строго лучшее. Если
    улучшений нет, делается случайный **нейтральный** шаг (точность та же) —
    иначе нельзя добавить узел, который сам по себе ещё ни к чему не подключён.
    """
    tt = TruthTable(task)
    start = _to_boolean_basis(graph)
    limit = max_nodes if max_nodes is not None else len(start[0]) + 8

    best, best_correct, total_steps = start, tt.score(*start), 0
    for r in range(restarts):
        cand, correct, steps = _search(tt, start, max_steps, limit, seed + r, plateau)
        total_steps += steps
        if correct > best_correct:
            best, best_correct = cand, correct
        if best_correct >= tt.n_rows:
            break

    b_ops, b_src, b_outs = best
    result = HardGraph(
        task=task.name,
        n_inputs=task.n_inputs,
        n_outputs=task.n_outputs,
        ops=list(b_ops),
        src=[(int(a), int(b)) for a, b in b_src],
        outputs=[int(o) for o in b_outs],
        meta=dict(graph.meta),
    ).pruned()

    x, _ = task.table()
    return RefineResult(
        graph=result,
        accuracy=best_correct / tt.n_rows,
        correct=best_correct,
        total=tt.n_rows,
        steps=total_steps,
        start_accuracy=tt.score(*start) / tt.n_rows,
        boolean=boolean_domain(result, x),
    )
