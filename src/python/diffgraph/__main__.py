"""CLI: обучение топологии и экспорт графа для C++-движка.

Примеры
-------
::

    python -m diffgraph --task parity4
    python -m diffgraph --all --restarts 24
    python -m diffgraph --task add2 --nodes 12 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .export import HardGraph
from .tasks import TASKS, Task, get_task
from .train import TrainConfig, TrainResult, train

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m diffgraph",
        description="Обучение топологии графа (Gumbel-Softmax) и экспорт для C++.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", action="append", default=None,
                   help=f"какая задача: {', '.join(sorted(TASKS))} (можно несколько раз)")
    p.add_argument("--all", action="store_true", help="все задачи")
    p.add_argument("--nodes", type=int, default=8, help="узлов в графе (по умолчанию 8)")
    p.add_argument("--steps", type=int, default=3000, help="шагов обучения на рестарт")
    p.add_argument("--restarts", type=int, default=8, help="число рестартов")
    p.add_argument("--lr", type=float, default=0.15, help="скорость обучения")
    p.add_argument("--tau-start", type=float, default=2.0, help="начальная температура")
    p.add_argument("--tau-end", type=float, default=0.1, help="конечная температура")
    p.add_argument("--range-weight", type=float, default=0.05,
                   help="вес штрафа за значения вне [0,1]")
    p.add_argument("--batch", type=int, default=8,
                   help="размер мини-батча (0 = полный батч)")
    p.add_argument("--soft-out", action=argparse.BooleanOptionalAction, default=True,
                   help="мягкий выход: даёт градиент «холодным» узлам (--no-soft-out чтобы выключить)")
    p.add_argument("--refine", type=int, default=120,
                   help="шагов дискретной добивки после градиента (0 = выключить)")
    p.add_argument("--hard-start", type=float, default=0.3,
                   help="доля шагов до включения straight-through (0..1)")
    p.add_argument("--seed", type=int, default=0, help="начальное зерно")
    p.add_argument("--out-dir", type=Path, default=GRAPHS_DIR, help="куда сохранять графы")
    p.add_argument("--json", action="store_true", help="записать сводку в results/")
    p.add_argument("--quiet", action="store_true", help="без лога рестартов")
    return p.parse_args(argv)


def print_truth_table(task: Task, graph: HardGraph) -> bool:
    """Напечатать таблицу истинности и вернуть True, если всё сошлось."""
    x, y = task.table()
    ok = True
    header = " ".join(f"in{i}" for i in range(task.n_inputs))
    print(f"      {header:<{max(12, task.n_inputs * 4)}}  ожид  получ")
    for row_x, row_y in zip(x, y):
        got = graph.simulate(row_x)
        want = [int(v) for v in row_y]
        match = got == want
        ok = ok and match
        bits = "".join(str(int(v)) for v in row_x)
        print(
            f"      {bits:<{max(12, task.n_inputs * 4)}}  "
            f"{''.join(map(str, want)):<4}  "
            f"{''.join(map(str, got)):<5}  {'ok' if match else 'FAIL'}"
        )
    return ok


def report(task: Task, result: TrainResult, out_dir: Path) -> dict:
    best = result.best
    graph = best.graph

    print(f"\n  Найденная структура ({graph.n_nodes} узлов, {graph.edge_count()} рёбер):")
    for line in graph.describe().splitlines():
        print(f"      {line}")

    print("\n  Таблица истинности:")
    table_ok = print_truth_table(task, graph)

    acc = best.acc_examples * 100.0
    print(f"\n  Точность на всех {task.n_examples} входах: {acc:.2f}%")
    print(f"  Совпадение с целочисленной семантикой C++: {'да' if best.bit_exact else 'НЕТ'}")
    print(f"  Таблица истинности проверена заново: {'да' if table_ok else 'НЕТ'}")

    # Нерешённый граф не должен затирать решённый: иначе эксперимент
    # «--refine 0» молча испортит results/graphs/<task>.json, и следующий
    # запуск C++ честно покажет FAIL там, где всё работало.
    solved = best.acc_examples >= 1.0 and best.bit_exact and table_ok
    name = f"{task.name}.json" if solved else f"{task.name}.unsolved.json"
    path = out_dir / name
    graph.save(path)
    print(f"  Экспорт: {path}")
    if not solved:
        print(f"  ВНИМАНИЕ: задача не решена — сохранено под именем {name},")
        print(f"            прежний {task.name}.json не тронут.")

    summary = result.summary()
    summary["export"] = str(path)
    summary["truth_table_ok"] = table_ok
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Консоль Windows по умолчанию не в UTF-8 — иначе русский текст в логе
    # превращается в кракозябры при перенаправлении вывода в файл.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    torch.set_num_threads(1)  # тензоры крошечные, потоки только мешают
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.all:
        names = sorted(TASKS)
    elif args.task:
        names = args.task
    else:
        names = ["parity4"]
    tasks = [get_task(n) for n in names]

    cfg = TrainConfig(
        n_nodes=args.nodes,
        steps=args.steps,
        restarts=args.restarts,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        range_weight=args.range_weight,
        hard_start=args.hard_start,
        batch=args.batch,
        soft_out=args.soft_out,
        refine_steps=args.refine,
        verbose=not args.quiet,
    )

    print("=" * 70)
    print("  ДИФФЕРЕНЦИРУЕМЫЙ ГРАФ — обучение топологии (Gumbel-Softmax)")
    print("=" * 70)
    print(f"  torch {torch.__version__}, потоков: {torch.get_num_threads()}")
    print(f"  узлов: {cfg.n_nodes}, шагов: {cfg.steps}, рестартов: {cfg.restarts}")

    summaries: dict[str, dict] = {}
    t_all = time.perf_counter()

    for task in tasks:
        print(f"\n{'=' * 70}")
        print(f"  Задача: {task.name} — {task.description}")
        print(f"  входов {task.n_inputs}, выходов {task.n_outputs}, примеров {task.n_examples}")
        print(f"{'=' * 70}")

        t0 = time.perf_counter()
        result = train(task, cfg, seed0=args.seed)
        elapsed = time.perf_counter() - t0
        print(f"  обучение заняло {elapsed:.2f} c, рестартов использовано: {len(result.history)}")
        summaries[task.name] = report(task, result, args.out_dir)

    total = time.perf_counter() - t_all
    print(f"\n{'=' * 70}")
    print(f"  Готово за {total:.2f} c")
    print(f"{'=' * 70}")

    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "diff_graph_training.json"
        payload = {
            "engine": "torch-diffgraph",
            "torch_version": torch.__version__,
            "config": {
                "n_nodes": cfg.n_nodes,
                "steps": cfg.steps,
                "restarts": cfg.restarts,
                "lr": cfg.lr,
                "hard_start": cfg.hard_start,
            },
            "tasks": summaries,
            "seconds": round(total, 3),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  Сводка: {out}")

    # Код возврата — сигнал для скриптов: все задачи должны быть решены.
    all_ok = all(s["accuracy_examples"] >= 1.0 and s["bit_exact"] for s in summaries.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
