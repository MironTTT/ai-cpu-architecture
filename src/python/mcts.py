import re
import json
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from llama_cpp import Llama

# ---------- Пути ----------
# Скрипт лежит в src/python/, поэтому корень проекта — на два уровня выше (../../).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Настройки ----------
N_TASKS = 30
COMPLETION_BUDGET = 1200
N_BRANCHES = 3
N_THREADS = 4
UCB_C = 1.41
MAX_STEP_TOKENS = 80
MAX_DEPTH = 8
STEP_TEMPERATURE = 0.8
MAX_ITERATIONS = 40 

print("Загружаю модель...")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=N_THREADS, verbose=False)
print("Модель загружена.\n")


# ---------- Узел дерева ----------
@dataclass
class Node:
    text: str
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    answer: Optional[str] = None

    def ucb(self, c=UCB_C):
        if self.visits == 0:
            return float('inf')
        exploit = self.total_value / self.visits
        explore = c * math.sqrt(math.log(max(self.parent.visits, 1)) / self.visits)
        return exploit + explore


# ---------- Утилиты ----------
def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"[Оо]твет[:\s]*\*?\*?\s*([-+]?\d[\d,\.]*)", text)
    if not m:
        return None
    return m.group(1).replace(",", "").replace(".", "").strip()


def last_line(text: str) -> str:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text


def arithmetic_bonus(text: str) -> Optional[float]:
    """Ищет 'a op b = c' и проверяет. Возвращает долю верных (0..1) или None."""
    pattern = r"(\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(\d+(?:[.,]\d+)?)\s*=\s*(\d+(?:[.,]\d+)?)"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    correct = 0
    for a, op, b, c in matches:
        try:
            a_, b_, c_ = float(a.replace(",", ".")), float(b.replace(",", ".")), float(c.replace(",", "."))
            if op == "/" and b_ == 0:
                continue
            expected = eval(f"{a_} {op} {b_}")
            if abs(expected - c_) < 0.01 * max(1, abs(expected)):
                correct += 1
        except Exception:
            pass
    return correct / len(matches)


def build_step_prompt(problem: str, partial: str) -> str:
    if not partial:
        return (
            f"{problem}\n\n"
            "Реши задачу пошагово. Начни с первого шага. "
            "Когда получишь итоговый ответ, напиши 'Ответ: <число>'."
        )
    return (
        f"{problem}\n\n"
        f"Решение до сих пор:\n{partial}\n\n"
        "Продолжи решение следующим шагом. Если ответ уже получен — напиши 'Ответ: <число>'."
    )


# ---------- Генерация и оценка ----------
def generate_step(prompt: str):
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_STEP_TOKENS,
        temperature=STEP_TEMPERATURE,
    )
    text = out["choices"][0]["message"]["content"].strip()
    usage = out["usage"]
    return text, usage["completion_tokens"], usage["total_tokens"]


def self_evaluate(problem: str, step_text: str):
    prompt = (
        f"Задача: {problem}\n\n"
        f"Последний шаг рассуждения:\n{step_text}\n\n"
        "Насколько этот шаг логически верный и ведёт к правильному ответу? "
        "Ответь ТОЛЬКО числом от 0.0 до 1.0:"
    )
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    text = out["choices"][0]["message"]["content"].strip()
    usage = out["usage"]
    m = re.search(r"(\d+\.?\d*)", text)
    score = float(m.group(1)) if m else 0.5
    score = max(0.0, min(1.0, score))
    return score, usage["completion_tokens"], usage["total_tokens"]


# ---------- MCTS ----------
def backprop(node: Node, value: float):
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent


def best_path(root: Node) -> Node:
    node = root
    while node.children:
        terminals = [c for c in node.children if c.is_terminal and c.answer]
        if terminals:
            node = max(terminals, key=lambda c: c.visits)
        else:
            node = max(node.children, key=lambda c: c.visits)
    return node


def mcts_solve(problem: str):
    completion_used = 0
    total_used = 0
    root = Node(text="")
    iterations = 0

    while completion_used < COMPLETION_BUDGET and iterations < MAX_ITERATIONS:
        iterations += 1

        # ---- SELECT ----
        node = root
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda c: c.ucb())

        if node.is_terminal:
            break   # нашли терминальный на лучшем пути

        if node.depth >= MAX_DEPTH:
            node.is_terminal = True
            node.answer = extract_answer(node.text)
            score = 0.5 if node.answer else 0.1
            backprop(node, score)
            continue

        # ---- EXPAND ----
        prompt = build_step_prompt(problem, node.text)
        new_children = []
        for _ in range(N_BRANCHES):
            if completion_used >= COMPLETION_BUDGET:
                break
            try:
                step_text, comp, tot = generate_step(prompt)
            except Exception:
                continue
            completion_used += comp
            total_used += tot

            new_text = (node.text + "\n" + step_text).strip() if node.text else step_text
            child = Node(text=new_text, parent=node, depth=node.depth + 1)

            ans = extract_answer(new_text)
            if ans and "Ответ" in step_text:
                child.is_terminal = True
                child.answer = ans

            node.children.append(child)
            new_children.append(child)

        if not new_children:
            break

        # ---- EVAL ----
        for child in new_children:
            if completion_used >= COMPLETION_BUDGET:
                break
            try:
                eval_score, comp, tot = self_evaluate(problem, last_line(child.text))
            except Exception:
                eval_score, comp, tot = 0.5, 0, 0
            completion_used += comp
            total_used += tot

            arith = arithmetic_bonus(child.text)
            if arith is not None:
                score = 0.5 * eval_score + 0.5 * arith
            else:
                score = eval_score

            if child.is_terminal and child.answer:
                score = max(score, 0.85)   # бонус за наличие ответа

            backprop(child, score)

    best = best_path(root)
    return best.text, best.answer, completion_used, total_used, iterations


# ---------- Основной прогон ----------
def main():
    from datasets import load_dataset

    print(f"Загружаю GSM8K, первые {N_TASKS} задач...")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_TASKS))

    # Загружаем baseline для сравнения
    try:
        with open(RESULTS_DIR / "baseline_results.json", "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_by_i = {r["i"]: r for r in baseline_data["results"]}
    except FileNotFoundError:
        baseline_by_i = {}
        print("WARNING: baseline_results.json не найден — сравнение не будет\n")

    correct = 0
    total_completion = 0
    total_tokens_all = 0
    results = []

    t0 = time.time()
    for i, ex in enumerate(ds, 1):
        problem = ex["question"]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")

        best_text, pred, comp_used, tot_used, iters = mcts_solve(problem)
        ok = (pred == gold)
        if ok:
            correct += 1
        total_completion += comp_used
        total_tokens_all += tot_used

        results.append({
            "i": i,
            "problem": problem,
            "gold": gold,
            "pred": pred,
            "best_text": best_text,
            "correct": ok,
            "completion_tokens": comp_used,
            "total_tokens": tot_used,
            "iterations": iters,
        })

        baseline_ok = baseline_by_i.get(i, {}).get("correct", None)
        baseline_str = "" if baseline_ok is None else f" baseline={'✓' if baseline_ok else '✗'}"
        print(
            f"[{i}/{N_TASKS}] mcts={'✓' if ok else '✗'}{baseline_str} "
            f"comp={comp_used} iter={iters} acc={correct/i:.3f} elapsed={time.time()-t0:.0f}s"
        )

    # ---- Итоги ----
    accuracy = correct / len(ds)
    avg_comp = total_completion / len(ds)
    avg_total = total_tokens_all / len(ds)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ MCTS")
    print("=" * 60)
    print(f"Accuracy:                  {accuracy:.3f}  ({correct}/{len(ds)})")
    print(f"Ср. completion-токенов:    {avg_comp:.0f} / задача")
    print(f"Ср. total-токенов:         {avg_total:.0f} / задача")
    print(f"Время:                     {time.time() - t0:.0f} сек")

    # Сравнение с baseline на тех же задачах
    if baseline_by_i:
        n = min(len(ds), max(baseline_by_i.keys()))
        base_correct = sum(1 for i in range(1, n + 1) if baseline_by_i.get(i, {}).get("correct", False))
        base_acc = base_correct / n
        print("\n" + "-" * 60)
        print(f"Baseline на тех же {n} задачах: {base_acc:.3f}  ({base_correct}/{n})")
        print(f"Δ accuracy (MCTS - baseline): {(accuracy - base_acc) * 100:+.1f} п.п.")
        print("-" * 60)

    OUTPUT_PATH = RESULTS_DIR / "mcts_results.json"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_tasks": len(ds),
                "accuracy": round(accuracy, 4),
                "avg_completion_tokens": round(avg_comp, 1),
                "avg_total_tokens": round(avg_total, 1),
                "completion_budget": COMPLETION_BUDGET,
                "n_branches": N_BRANCHES,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()