#!/usr/bin/env python
"""Collect eval/results/*.summary.json (+ SBS results) into eval/RESULTS.md.

  python scripts/compare.py                       # all systems
  python scripts/compare.py yandex-base yandex-ft  # chosen systems, in this order
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "eval/results")

COLUMNS = [
    ("structure_ok", "Структура рецепта", True),
    ("house_format", "Целевой формат", None),
    ("dish/must_have_recall", "Ключевые ингредиенты (классика)", True),
    ("ingredients/requested_coverage", "Использованы продукты из запроса", True),
    ("ingredient_usage", "Ингредиенты используются в шагах", True),
    ("quantities_share", "Есть количества", True),
    ("chatter", "«Болтовня» форума", False),
    ("repetition", "Повторы", False),
    ("truncated", "Обрыв ответа", False),
    ("n_steps", "Шагов", None),
    ("chars", "Длина, симв.", None),
    ("retrieval_quality", "RAG: найден подходящий рецепт", None),
]


def fmt(v, key):
    if v is None:
        return "—"
    if key in ("n_steps", "chars"):
        return f"{v:.0f}" if key == "chars" else f"{v:.1f}"
    return f"{100 * v:.0f}%"


def main():
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-len(".summary.json")]
                                   for p in glob.glob(os.path.join(RES, "*.summary.json")))
    summaries = []
    for n in names:
        with open(os.path.join(RES, f"{n}.summary.json"), encoding="utf-8") as f:
            summaries.append(json.load(f))

    from collections import Counter

    with open(os.path.join(ROOT, "eval/prompts.jsonl"), encoding="utf-8") as f:
        c = Counter(json.loads(l)["type"] for l in f if l.strip())
    lines = ["# Результаты оценки", "",
             f"Набор: `eval/prompts.jsonl` ({sum(c.values())} запросов: {c['dish']} классических блюд, "
             f"{c['ingredients']} наборов продуктов, {c['category']} категорий), жадное декодирование. "
             "↑ — больше лучше, ↓ — меньше лучше.", ""]
    head = "| Система | " + " | ".join(f"{t} {'↑' if up else '↓' if up is False else ''}".strip()
                                       for _, t, up in COLUMNS) + " |"
    lines += [head, "|" + "---|" * (len(COLUMNS) + 1)]
    for s in summaries:
        cells = []
        for key, _, up in COLUMNS:
            vals = [x.get(key) for x in summaries if x.get(key) is not None]
            v = s.get(key)
            best = up is not None and v is not None and len(set(vals)) > 1 and v == (max(vals) if up else min(vals))
            cells.append(f"**{fmt(v, key)}**" if best else fmt(v, key))
        lines.append(f"| {s['system']} | " + " | ".join(cells) + " |")

    sbs = sorted(glob.glob(os.path.join(ROOT, "eval/sbs_*_results.json")))
    if sbs:
        lines += ["", "## Слепое сравнение (SBS)", "", "| Пара | Побед A | Ничьих | Побед B | Win rate B |", "|---|---|---|---|---|"]
        for p in sbs:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            n = r["a_wins"] + r["ties"] + r["b_wins"]
            wr = (r["b_wins"] + 0.5 * r["ties"]) / n if n else 0
            lines.append(f"| {r['a']} vs **{r['b']}** | {r['a_wins']} | {r['ties']} | {r['b_wins']} | {100 * wr:.0f}% |")

    out = os.path.join(ROOT, "eval/RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
