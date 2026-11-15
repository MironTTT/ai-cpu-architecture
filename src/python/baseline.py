import re
import json
import time
from pathlib import Path

from llama_cpp import Llama
from datasets import load_dataset

# ---------- Пути ----------
# Скрипт лежит в src/python/, поэтому корень проекта — на два уровня выше (../../).
# Пути считаются от __file__, а не от текущей папки, поэтому скрипт запускается откуда угодно.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Настройки ----------
N_TASKS = 300          # сколько задач прогнать (300 = ~30-60 мин на CPU)
MAX_TOKENS = 800       # бюджет на ответ
N_THREADS = 4          # ядра CPU

# ---------- Загрузка модели ----------
print("Загружаю модель...")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=N_THREADS, verbose=False)
print("Модель загружена.\n")

# ---------- Утилиты ----------
def extract_answer(text: str):
    """Достаёт число после 'Ответ:' (или последнее число в тексте)."""
    m = re.search(r"[Оо]твет[:\s]*\*?\*?\s*([-+]?\d[\d,\.]*)", text)
    if not m:
        nums = re.findall(r"[-+]?\d[\d,\.]*", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    # очистка: убрать запятые и точки на конце
    return m_val.replace(",", "").replace(".", "").strip()

def solve(problem: str):
    """Один прогон модели на задачу. Возвращает (текст ответа, tokens)."""
    prompt = (
        f"{problem}\n\n"
        "Реши пошагово. В самом конце напиши ровно в таком формате:\n"
        "Ответ: <число>"
    )
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,   # детерминированно, для честного сравнения
    )
    text = out["choices"][0]["message"]["content"]
    tokens = out["usage"]["total_tokens"]
    return text, tokens

# ---------- Прогон ----------
print(f"Загружаю GSM8K, беру первые {N_TASKS} задач...")
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_TASKS))
print(f"Задач: {len(ds)}\n")

correct = 0
total_tokens = 0
results = []   # сюда сохраним каждый пример

t0 = time.time()
for i, ex in enumerate(ds, 1):
    problem = ex["question"]
    gold = ex["answer"].split("####")[-1].strip().replace(",", "")

    pred_text, tok = solve(problem)
    pred = extract_answer(pred_text)

    is_correct = (pred == gold)
    if is_correct:
        correct += 1
    total_tokens += tok

    results.append({
        "i": i,
        "problem": problem,
        "gold": gold,
        "pred_text": pred_text,
        "pred": pred,
        "correct": is_correct,
        "tokens": tok,
    })

    if i % 10 == 0:
        elapsed = time.time() - t0
        acc_so_far = correct / i
        print(f"[{i}/{N_TASKS}] acc={acc_so_far:.3f}  avg_tok={total_tokens/i:.0f}  elapsed={elapsed:.0f}s")

# ---------- Итоги ----------
accuracy = correct / len(ds)
avg_tokens = total_tokens / len(ds)

summary = {
    "model": str(MODEL_PATH),
    "n_tasks": len(ds),
    "accuracy": round(accuracy, 4),
    "avg_tokens_per_task": round(avg_tokens, 1),
    "total_tokens": total_tokens,
    "elapsed_sec": round(time.time() - t0, 1),
}

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТ BASELINE")
print("=" * 50)
print(f"Accuracy:        {accuracy:.3f}  ({correct}/{len(ds)})")
print(f"Ср. токенов:     {avg_tokens:.0f} / задача")
print(f"Всего токенов:   {total_tokens}")
print(f"Время:           {summary['elapsed_sec']} сек")

# Сохраняем результаты на диск
OUTPUT_PATH = RESULTS_DIR / "baseline_results.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

print(f"\nРезультаты сохранены в {OUTPUT_PATH}")