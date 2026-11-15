"""Быстрая проверка: загружается ли GGUF-модель и отвечает ли она.

Запуск из папки src/python/:
    python test_model.py
"""

from pathlib import Path

from llama_cpp import Llama

# ---------- Пути ----------
# Скрипт лежит в src/python/, поэтому корень проекта — на два уровня выше (../../).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

if not MODEL_PATH.exists():
    raise SystemExit(f"Модель не найдена: {MODEL_PATH}")

# Загружаем модель из файла
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,        # размер контекста
    n_threads=4,       # сколько ядер CPU использовать
    verbose=False,
)

# Задаём вопрос
ответ = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Сколько будет 17 * 23? Ответь пошагово."}
    ],
    max_tokens=300,
)

# Печатаем ответ модели
print(ответ["choices"][0]["message"]["content"])
