"""Обучение топологии: цикл, расписание температуры, отбор лучшего кандидата.

Что происходит внутри
---------------------
1. Логиты структуры инициализируются случайно (симметрию ломает шум).
2. На каждом шаге Gumbel-Softmax сэмплирует операцию и два источника на узел.
3. Температура ``tau`` падает по косинусу: в начале выбор мягкий, в конце —
   почти жёсткий.
4. После ``hard_start`` доли шагов включается straight-through режим: вперёд
   идёт уже настоящий дискретный граф, а градиент считается по мягкому.
5. Каждые ``eval_every`` шагов граф «затвердевается» (argmax) и проверяется на
   полной таблице истинности. Лучшее состояние запоминается.

Отбор идёт по дискретной точности, а не по значению loss: нам нужен граф,
который работает после затвердевания, а не мягкая аппроксимация.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .export import HardGraph
from .model import DifferentiableGraph, OPS
from .refine import bit_accuracy, refine
from .tasks import Task


@dataclass
class TrainConfig:
    n_nodes: int = 8
    steps: int = 3000
    restarts: int = 8
    lr: float = 0.15
    tau_start: float = 2.0       # высокая стартовая температура = исследование
    tau_end: float = 0.1
    hard_start: float = 0.3      # доля шагов, после которой включается ST-режим
    range_weight: float = 0.05   # штраф за значения вне [0, 1]
    batch: int = 8               # 0 = полный батч; иначе мини-батч (шум в градиенте)
    soft_out: bool = True        # мягкий выход: лечит «холодные» узлы
    refine_steps: int = 120      # 0 = без дискретной добивки
    eval_every: int = 25
    verbose: bool = True


@dataclass
class Candidate:
    """Один обученный вариант структуры."""

    graph: HardGraph
    acc_examples: float = 0.0    # доля примеров, где верны ВСЕ выходные биты
    acc_bits: float = 0.0        # доля верных отдельных битов
    bit_exact: bool = False      # совпадает ли float-граф с целочисленным C++
    loss: float = float("inf")
    seed: int = -1
    step: int = 0
    seconds: float = 0.0
    acc_gradient: float | None = None   # точность ДО дискретной добивки
    boolean: bool = True                # все значения остаются в {0, 1}

    @property
    def solved(self) -> bool:
        return self.acc_examples >= 1.0 and self.bit_exact

    @property
    def refined(self) -> bool:
        return self.acc_gradient is not None

    def key(self) -> tuple[float, int, int]:
        """Критерий отбора: точность, затем совместимость с C++, затем размер."""
        return (self.acc_examples, int(self.bit_exact), -self.graph.n_nodes)


@dataclass
class TrainResult:
    task: Task
    best: Candidate
    history: list[Candidate] = field(default_factory=list)
    config: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> dict[str, Any]:
        b = self.best
        return {
            "task": self.task.name,
            "accuracy_examples": b.acc_examples,
            "accuracy_bits": b.acc_bits,
            "accuracy_gradient": b.acc_gradient,
            "refined": b.refined,
            "boolean_domain": b.boolean,
            "bit_exact": b.bit_exact,
            "nodes": b.graph.n_nodes,
            "edges": b.graph.edge_count(),
            "ops": list(b.graph.ops),
            "seed": b.seed,
            "restarts_used": len(self.history),
            "seconds_to_best": round(b.seconds, 3),
        }


# ---------------------------------------------------------------------------
def _hard_graph(model: DifferentiableGraph, task: Task) -> HardGraph:
    """Затвердеть и отрезать недостижимые узлы."""
    src, ops, outs = model.discrete_structure()
    graph = HardGraph.from_structure(
        task=task.name,
        n_inputs=task.n_inputs,
        n_outputs=task.n_outputs,
        src=src.tolist(),
        ops=ops.tolist(),
        outputs=outs.tolist(),
    )
    return graph.pruned()


def evaluate(model: DifferentiableGraph, task: Task) -> Candidate:
    """Дискретная точность + проверка совпадения с целочисленной семантикой."""
    x_np, y_np = task.table()
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np)

    out = model.discrete_outputs(x)                 # [N, O], float
    graph = _hard_graph(model, task)

    # Тот же граф, но на целых числах — ровно так его исполнит C++.
    int_rows = [graph.simulate(row) for row in x_np]
    int_out = torch.tensor(int_rows, dtype=torch.float32)

    bit_exact = bool(torch.allclose(out, int_out, atol=1e-5))
    pred = (out >= 0.5).int()
    acc_bits = (pred == y).float().mean().item()
    acc_examples = (pred == y).all(dim=1).float().mean().item()

    return Candidate(
        graph=graph,
        acc_examples=acc_examples,
        acc_bits=acc_bits,
        bit_exact=bit_exact,
    )


# ---------------------------------------------------------------------------
def train_once(task: Task, cfg: TrainConfig, seed: int) -> tuple[Candidate, Candidate]:
    """Один прогон обучения. Возвращает (лучший на этом прогоне, финальный)."""
    torch.manual_seed(seed)
    model = DifferentiableGraph(task.n_inputs, task.n_outputs, cfg.n_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.steps), eta_min=cfg.lr * 0.02
    )

    x_np, y_np = task.table()
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()

    best: Candidate | None = None
    last: Candidate | None = None
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        t = step / max(1, cfg.steps - 1)
        tau = cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1.0 + math.cos(math.pi * t))
        hard = t >= cfg.hard_start

        # Мини-батч даёт шум в градиенте: на полном батче из 16 примеров
        # оптимизация детерминирована и легко залипает в локальном минимуме.
        if 0 < cfg.batch < x.shape[0]:
            idx = torch.randint(0, x.shape[0], (cfg.batch,))
            xb, yb = x[idx], y[idx]
        else:
            xb, yb = x, y

        out, penalty = model(xb, tau=tau, hard=hard, noise=True, soft_out=cfg.soft_out)
        loss = F.mse_loss(out, yb) + cfg.range_weight * penalty
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            cand = evaluate(model, task)
            cand.loss = float(loss.detach())
            cand.seed = seed
            cand.step = step
            cand.seconds = time.perf_counter() - t0
            last = cand
            if best is None or cand.key() > best.key():
                best = cand
            if cand.solved:
                return cand, cand

    assert best is not None and last is not None
    return best, last


# ---------------------------------------------------------------------------
def train(
    task: Task,
    cfg: TrainConfig | None = None,
    seed0: int = 0,
    log=print,
) -> TrainResult:
    """Обучить структуру с нескольких рестартов и вернуть лучшую."""
    cfg = cfg or TrainConfig()
    history: list[Candidate] = []
    best: Candidate | None = None

    for r in range(cfg.restarts):
        seed = seed0 + r
        cand, final = train_once(task, cfg, seed)
        history.append(cand)

        if cfg.verbose:
            mark = "✓" if cand.solved else " "
            log(
                f"    рестарт {r + 1:2d}/{cfg.restarts}  seed={seed:<3d}  "
                f"точность={cand.acc_examples * 100:6.2f}%  "
                f"бит-в-бит={'да ' if cand.bit_exact else 'нет'}  "
                f"узлов={cand.graph.n_nodes:2d}  {mark}"
            )

        if best is None or cand.key() > best.key():
            best = cand
        if cand.solved:
            break

    assert best is not None

    # Дискретная добивка: градиент двигает логиты узлов независимо и не умеет
    # делать согласованные изменения. Локальный поиск по правкам структуры
    # закрывает последние проценты — это стандартный шаг после DARTS-подобного
    # поиска, и он явно отделён в отчёте.
    if not best.solved and cfg.refine_steps > 0:
        before = best.acc_examples
        rr = refine(best.graph, task, max_steps=cfg.refine_steps, seed=seed0)
        if cfg.verbose:
            log(
                f"    добивка структуры: {before * 100:.2f}% -> {rr.accuracy * 100:.2f}%  "
                f"({rr.steps} шагов)"
            )
        best = Candidate(
            graph=rr.graph,
            acc_examples=rr.accuracy,
            acc_bits=bit_accuracy(rr.graph, *task.table()),
            bit_exact=True,          # граф дискретный, семантика — целочисленная
            loss=best.loss,
            seed=best.seed,
            step=best.step,
            seconds=best.seconds,
            acc_gradient=before,
            boolean=rr.boolean,
        )

    return TrainResult(task=task, best=best, history=history, config=cfg)
