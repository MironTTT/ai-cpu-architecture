"""
Best-of-N + арифметический верификатор для GSM8K.
- N полных решений на задачу (T=0.7 для диверсити)
- Верификатор проверяет "a op b = c" на лету
- Селекция: идеальная арифметика → majority vote → fallback
"""
import re
import json
import time
from collections import Counter
from pathlib import Path

from llama_cpp import Llama
from datasets import load_dataset

# ---------- Пути ----------
# Скрипт лежит в src/python/, поэтому корень проекта — на два уровня выше (../../).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Настройки ----------
N_TASKS = 30
N_SAMPLES = 5
MAX_TOKENS = 500
N_THREADS = 4
TEMPERATURE = 0.7

print("Загружаю модель...")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=N_THREADS, verbose=False)
print("Модель загружена.\n")


# ---------- Утилиты ----------
def extract_answer(text):
    m = re.search(r"[Оо]твет[:\s]*\*?\*?\s*([-+]?\d[\d,\.]*)", text)
    if not m:
        return None
    return m.group(1).replace(",", "").replace(".", "").strip()


def arithmetic_score(text):
    """Возвращает (correct_count, total_count). Если нет 'a op b = c' — (0, 0)."""
    pattern = r"(\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(\d+(?:[.,]\d+)?)\s*=\s*(\d+(?:[.,]\d+)?)"
    matches = re.findall(pattern, text)
    if not matches:
        return 0, 0
    correct = 0
    for a, op, b, c in matches:
        try:
            a_ = float(a.replace(",", "."))
            b_ = float(b.replace(",", "."))
            c_ = float(c.replace(",", "."))
            if op == "/" and b_ == 0:
                continue
            expected = eval(f"{a_} {op} {b_}")
            if abs(expected - c_) < 0.01 * max(1, abs(expected)):
                correct += 1
        except Exception:
            pass
    return correct, len(matches)


# ---------- Генерация ----------
def solve_full(problem):
    prompt = (
        f"{problem}\n\n"
        "Реши пошагово. В самом конце напиши ровно в таком формате:\n"
        "Ответ: <число>"
    )
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    text = out["choices"][0]["message"]["content"]
    usage = out["usage"]
    return text, usage["completion_tokens"], usage["total_tokens"]


# ---------- Селекция ----------
def select_best(samples):
    """
    samples: list of {text, answer, arith_ok, arith_total}
    Возвращает (predicted_answer, reason)
    """
    # 1. Приоритет — образцы с идеальной арифметикой и ответом
    perfect = [s for s in samples if s["arith_total"] > 0
               and s["arith_ok"] == s["arith_total"] and s["answer"]]

    if perfect:
        counts = Counter(s["answer"] for s in perfect)
        best_answer, n = counts.most_common(1)[0]
        if len(counts) == 1:
            return best_answer, "unanimous_perfect"
        return best_answer, "majority_perfect"

    # 2. Fallback — максимальная доля верной арифметики
    with_answers = [s for s in samples if s["answer"]]
    if not with_answers:
        return None, "no_answer"

    def ratio(s):
        if s["arith_total"] == 0:
            return 0.0
        return s["arith_ok"] / s["arith_total"]

    with_answers.sort(key=ratio, reverse=True)
    return with_answers[0]["answer"], "arith_partial"


# ---------- Основной прогон ----------
def main():
    print(f"Загружаю GSM8K, первые {N_TASKS} задач...")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_TASKS))

    try:
        with open(RESULTS_DIR / "baseline_results.json", "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_by_i = {r["i"]: r for r in baseline_data["results"]}
    except FileNotFoundError:
        baseline_by_i = {}
        print("WARNING: baseline_results.json не найден — сравнения не будет\n")

    correct = 0
    total_completion = 0
    total_tokens_all = 0
    results = []

    t0 = time.time()
    for i, ex in enumerate(ds, 1):
        problem = ex["question"]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")

        samples = []
        task_completion = 0
        task_total = 0

        for _ in range(N_SAMPLES):
            text, comp, tot = solve_full(problem)
            task_completion += comp
            task_total += tot
            ans = extract_answer(text)
            arith_ok, arith_total = arithmetic_score(text)
            samples.append({
                "text": text,
                "answer": ans,
                "arith_ok": arith_ok,
                "arith_total": arith_total,
            })

        pred, reason = select_best(samples)
        ok = (pred == gold)
        if ok:
            correct += 1
        total_completion += task_completion
        total_tokens_all += task_total

        # Диагностика
        answers = Counter(s["answer"] for s in samples if s["answer"])
        perfect_count = sum(1 for s in samples
                           if s["arith_total"] > 0 and s["arith_ok"] == s["arith_total"])

        results.append({
            "i": i,
            "problem": problem,
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "reason": reason,
            "n_unique_answers": len(answers),
            "perfect_arith_samples": perfect_count,
            "samples": samples,
            "completion_tokens": task_completion,
            "total_tokens": task_total,
        })

        baseline_ok = baseline_by_i.get(i, {}).get("correct", None)
        baseline_str = "" if baseline_ok is None else f" base={'✓' if baseline_ok else '✗'}"
        print(
            f"[{i}/{N_TASKS}] bon={'✓' if ok else '✗'}{baseline_str} "
            f"reason={reason} perfect={perfect_count}/{N_SAMPLES} "
            f"uniq={len(answers)} acc={correct/i:.3f} elapsed={time.time()-t0:.0f}s"
        )

    # ---- Итоги ----
    accuracy = correct / len(ds)
    avg_comp = total_completion / len(ds)
    avg_total = total_tokens_all / len(ds)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ BEST-OF-N")
    print("=" * 60)
    print(f"N samples:               {N_SAMPLES}")
    print(f"Accuracy:                {accuracy:.3f}  ({correct}/{len(ds)})")
    print(f"Ср. completion-токенов:  {avg_comp:.0f} / задача")
    print(f"Ср. total-токенов:       {avg_total:.0f} / задача")
    print(f"Время:                   {time.time() - t0:.0f} сек")

    if baseline_by_i:
        n = min(len(ds), max(baseline_by_i.keys()))
        base_correct = sum(1 for i in range(1, n + 1)
                          if baseline_by_i.get(i, {}).get("correct", False))
        base_acc = base_correct / n
        print("\n" + "-" * 60)
        print(f"Baseline на тех же {n} задачах: {base_acc:.3f}  ({base_correct}/{n})")
        print(f"Δ accuracy (bon - baseline): {(accuracy - base_acc) * 100:+.1f} п.п.")
        print("-" * 60)

    OUTPUT_PATH = RESULTS_DIR / "best_of_n_results.json"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_tasks": len(ds),
                "n_samples": N_SAMPLES,
                "accuracy": round(accuracy, 4),
                "avg_completion_tokens": round(avg_comp, 1),
                "avg_total_tokens": round(avg_total, 1),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()