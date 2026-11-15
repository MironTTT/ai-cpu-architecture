"""Дифференцируемый граф: узлы, «мягкие» операции и выбор структуры.

Идея
----
Жёсткий граф из C++-движка — это набор узлов, у каждого из которых есть тип
операции и два входных индекса. Здесь то же самое, но:

* вместо конкретной операции у узла вектор логитов ``op_logits[j]`` на все
  допустимые операции;
* вместо конкретных входов — логиты ``src_logits[j, k]`` на все источники
  (входы задачи + все предыдущие узлы);
* вместо конкретного выхода — логиты ``out_logits[o]`` на все узлы.

Из логитов Gumbel-Softmax сэмплирует one-hot вектор. Пока температура высокая
и включён режим ``hard=False``, это мягкая смесь («вероятности»), а к концу
обучения температура падает и выбор становится жёстким. В режиме ``hard=True``
вперёд идёт уже настоящий дискретный граф, а градиент — через
straight-through estimator, поэтому обучается ровно то, что потом исполнит C++.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Порядок обязан совпадать с opFromName() в src/cpp/graph_evolve.cpp.
OPS: tuple[str, ...] = ("AND", "OR", "XOR", "NAND", "ADD", "MUL", "COPY")
N_OPS: int = len(OPS)

# Индексы операций, которые считаются «арифметическими» и могут вывести
# значение за пределы [0, 1]. Используется только для диагностики.
ARITH_OPS = frozenset({"ADD", "MUL"})


def op_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Все операции сразу: ``a, b`` формы ``[B]`` -> ``[B, N_OPS]``.

    Это гладкие продолжения булевых вентилей (полиномы над вещественными
    числами). На входах 0/1 они дают ровно булев результат, а между 0 и 1 —
    непрерывную интерполяцию, по которой считается градиент:

    ====== ========================= ==========================
    опер.  гладкая форма             на битах
    ====== ========================= ==========================
    AND    ``a*b``                   0/1, как вентиль И
    OR     ``a+b-a*b``               0/1, как вентиль ИЛИ
    XOR    ``a+b-2*a*b``             0/1, как исключающее ИЛИ
    NAND   ``1-a*b``                 0/1, как И-НЕ
    ADD    ``a+b``                   настоящее сложение (может дать 2)
    MUL    ``a*b``                   произведение
    COPY   ``a``                     тождество
    ====== ========================= ==========================
    """
    ab = a * b
    return torch.stack(
        (
            ab,               # AND
            a + b - ab,       # OR
            a + b - 2.0 * ab, # XOR
            1.0 - ab,         # NAND
            a + b,            # ADD
            ab,               # MUL
            a,                # COPY
        ),
        dim=-1,
    )


def _one_hot(logits: torch.Tensor, tau: float, hard: bool, noise: bool) -> torch.Tensor:
    """Gumbel-Softmax (обучающий режим) или чистый argmax (оценка)."""
    if noise:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    idx = logits.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, idx, 1.0)


class DifferentiableGraph(nn.Module):
    """Граф с обучаемой топологией.

    Параметры
    ---------
    n_inputs, n_outputs : размерности задачи.
    n_nodes : сколько узлов доступно модели. Лишние узлы ей не мешают:
              после обучения недостижимые из выходов узлы отрезаются.
    """

    def __init__(self, n_inputs: int, n_outputs: int, n_nodes: int) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes

        slots = n_inputs + n_nodes

        # Логиты структуры. Небольшой шум, чтобы сломать симметрию.
        self.op_logits = nn.Parameter(torch.randn(n_nodes, N_OPS) * 0.3)
        self.src_logits = nn.Parameter(torch.randn(n_nodes, 2, slots) * 0.3)
        self.out_logits = nn.Parameter(torch.randn(n_outputs, slots) * 0.3)

        # Узел j видит только входы задачи и узлы 0..j-1 — это и есть DAG.
        mask = torch.zeros(n_nodes, slots, dtype=torch.bool)
        for j in range(n_nodes):
            mask[j, : n_inputs + j] = True
        self.register_buffer("src_mask", mask)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        noise: bool = True,
        soft_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Прогон батча ``x [B, n_inputs]`` -> ``(y [B, n_outputs], штраф)``.

        Штраф — мягкое наказание за значения вне [0, 1]: он удерживает граф в
        булевом домене и делает ненужными арифметические операции, из-за
        которых граф перестал бы совпадать с целочисленным C++-исполнителем.

        ``soft_out`` оставляет выходной слой мягким даже в жёстком режиме.
        Это лечит «холодные» узлы: узел, не подключённый ни к одному выходу,
        при жёстком выходе получает нулевой градиент и уже никогда не сможет
        подключиться. С мягким выходом у каждого узла есть ненулевой вес, а
        значит и градиент.
        """
        src_logits = self.src_logits.masked_fill(~self.src_mask.unsqueeze(1), -1e9)
        src_w = _one_hot(src_logits, tau, hard, noise)     # [N, 2, S]
        op_w = _one_hot(self.op_logits, tau, hard, noise)  # [N, K]

        vals = x
        penalties: list[torch.Tensor] = []
        for j in range(self.n_nodes):
            width = vals.shape[1]
            v0 = (src_w[j, 0, :width] * vals).sum(dim=-1)
            v1 = (src_w[j, 1, :width] * vals).sum(dim=-1)
            h = (op_w[j] * op_values(v0, v1)).sum(dim=-1)

            penalties.append(torch.relu(h - 1.0).pow(2) + torch.relu(-h).pow(2))
            vals = torch.cat((vals, h.unsqueeze(1)), dim=1)

        out_w = _one_hot(self.out_logits, tau, False if soft_out else hard, noise)
        y = vals @ out_w.t()                                 # [B, O]
        penalty = torch.cat(penalties).mean() if penalties else x.new_zeros(())
        return y, penalty

    # ------------------------------------------------------------------
    @torch.no_grad()
    def discrete_structure(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Затвердевшая структура: (src [N,2], ops [N], outputs [O])."""
        src_logits = self.src_logits.masked_fill(~self.src_mask.unsqueeze(1), -1e9)
        return (
            src_logits.argmax(dim=-1).cpu(),
            self.op_logits.argmax(dim=-1).cpu(),
            self.out_logits.argmax(dim=-1).cpu(),
        )

    @torch.no_grad()
    def discrete_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Выход жёсткого графа (без шума, без мягких смесей)."""
        y, _ = self.forward(x, tau=1.0, hard=True, noise=False)
        return y

    def structure_summary(self) -> str:
        """Человекочитаемая формула найденного графа."""
        src, ops, outs = self.discrete_structure()
        return describe(src, ops, outs, self.n_inputs)


def describe(
    src: torch.Tensor,
    ops: torch.Tensor,
    outs: torch.Tensor,
    n_inputs: int,
) -> str:
    """Текстовая запись графа: ``n4 = XOR(in0, in1)`` и т.п."""

    def name(idx: int) -> str:
        return f"in{idx}" if idx < n_inputs else f"n{idx}"

    lines: list[str] = []
    for j in range(src.shape[0]):
        lines.append(f"n{n_inputs + j} = {OPS[int(ops[j])]}({name(int(src[j, 0]))}, {name(int(src[j, 1]))})")
    for o in range(outs.shape[0]):
        lines.append(f"out{o} = {name(int(outs[o]))}")
    return "\n".join(lines)
