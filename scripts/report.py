#!/usr/bin/env python
"""Final report: bootstrap confidence intervals, paired effects, figures -> eval/REPORT.md + eval/figures/.

  python scripts/report.py                       # all systems found in eval/results, default effects
  python scripts/report.py --systems yandex-base yandex-ft qwen-base qwen-ft

Statistics
* Every metric is a mean over evaluation prompts; its 95% CI is a percentile bootstrap over prompts
  (B resamples with replacement).
* Effects ("what fine-tuning / RAG / the base model changes") are PAIRED: both systems answered the same
  prompts, so we resample prompt ids and take the difference of means on the same resample. This removes
  between-prompt variance and gives much tighter intervals than comparing two independent CIs.
* An effect is marked as significant when its 95% CI excludes 0. No multiple-comparison correction:
  read the markers as "worth attention", not as confirmatory tests.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "eval", "results")
FIG = os.path.join(ROOT, "eval", "figures")

# key, label, higher_is_better
METRICS = [
    ("must_have_recall", "Ключевые ингредиенты классики", True),
    ("requested_coverage", "Продукты из запроса использованы", True),
    ("ingredient_usage", "Ингредиенты используются в шагах", True),
    ("quantities_share", "У ингредиентов есть количества", True),
    ("chatter", "«Болтовня» форума", False),
    ("truncated", "Ответ оборван", False),
    ("structure_ok", "Структура рецепта", True),
]
FIG_METRICS = ["must_have_recall", "ingredient_usage", "quantities_share", "chatter", "truncated", "requested_coverage"]
SHORT = {"must_have_recall": "Ключевые ингредиенты классики", "requested_coverage": "Продукты из запроса",
         "ingredient_usage": "Ингредиенты используются в шагах", "quantities_share": "Количества у ингредиентов",
         "chatter": "«Болтовня» форума", "truncated": "Ответ оборван", "structure_ok": "Структура рецепта"}

DEFAULT_ORDER = ["yandex-base", "yandex-base-rag", "yandex-ft", "yandex-ft-rag",
                 "qwen-base", "qwen-base-rag", "qwen-ft", "qwen-ft-rag"]
EFFECTS = [
    ("Дообучение · YandexGPT", "yandex-ft", "yandex-base"),
    ("Дообучение · Qwen", "qwen-ft", "qwen-base"),
    ("RAG · YandexGPT", "yandex-base-rag", "yandex-base"),
    ("RAG · Qwen", "qwen-base-rag", "qwen-base"),
    ("RAG поверх дообучения · YandexGPT", "yandex-ft-rag", "yandex-ft"),
    ("RAG поверх дообучения · Qwen", "qwen-ft-rag", "qwen-ft"),
    ("YandexGPT − Qwen (обе дообучены)", "yandex-ft", "qwen-ft"),
]

# Reference palette (dataviz skill), validated: blue/orange pass CVD & contrast checks on the light surface.
C_YANDEX, C_QWEN = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


# ----------------------------------------------------------------------------- data & statistics
def load(name):
    with open(os.path.join(RES, f"{name}.jsonl"), encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


def values(rows, key):
    return {pid: float(r["metrics"][key]) for pid, r in rows.items() if key in r["metrics"]}


def boot_mean(vals: np.ndarray, rng, B: int):
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_paired(a: dict, b: dict, rng, B: int):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = np.array([a[i] - b[i] for i in ids])
    m, lo, hi = boot_mean(d, rng, B)
    return {"diff": m, "lo": lo, "hi": hi, "n": len(ids), "sig": lo > 0 or hi < 0}


def family(name):
    return "yandex" if name.startswith("yandex") else "qwen"


def style(name):
    color = C_YANDEX if family(name) == "yandex" else C_QWEN
    filled = "-ft" in name
    marker = "D" if name.endswith("-rag") else "o"
    return color, filled, marker


def pct(x):
    return f"{100 * x:.0f}%"


# ----------------------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })
    return plt


def legend_handles(plt):
    from matplotlib.lines import Line2D

    mk = lambda **kw: Line2D([], [], linestyle="none", markersize=7, markeredgewidth=1.6, **kw)
    return [
        mk(marker="o", color=C_YANDEX, label="YandexGPT-5-Lite-8B"),
        mk(marker="o", color=C_QWEN, label="Qwen2.5-7B"),
        mk(marker="o", color=INK2, markerfacecolor=SURFACE, label="исходная модель"),
        mk(marker="o", color=INK2, label="дообученная (LoRA)"),
        mk(marker="D", color=INK2, markerfacecolor=SURFACE, label="+ RAG"),
    ]


def fig_metrics(plt, table, systems, path, n_prompts):
    """Small multiples: one panel per metric, systems on y, mean ± 95% CI."""
    keys = [k for k in FIG_METRICS if any(table[s].get(k) for s in systems)]
    better = {k: hb for k, _, hb in METRICS}
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 0.42 * len(systems) * nrow + 1.4), sharey=True)
    fam = [family(s) for s in systems]
    seps = [i for i in range(1, len(systems)) if fam[i] != fam[i - 1]]
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(systems))[::-1]
    for ax, k in zip(axes, keys):
        for yi, s in zip(y, systems):
            st = table[s].get(k)
            if not st:
                continue
            color, filled, marker = style(s)
            m, lo, hi = st
            ax.plot([100 * lo, 100 * hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(100 * m, yi, marker=marker, markersize=7, color=color, markeredgewidth=1.6,
                    markerfacecolor=color if filled else SURFACE, linestyle="none", zorder=3)
        for i in seps:
            ax.axhline(len(systems) - i - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_title(SHORT[k] + ("  (↑ лучше)" if better[k] else "  (↓ лучше)"), fontsize=9.5)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", visible=False)
    for ax in axes[len(keys):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_yticks(y)
        ax.set_yticklabels(systems)
    fig.suptitle(f"Качество рецептов по системам: среднее и 95% доверительный интервал (бутстрэп по {n_prompts} запросам)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.legend(handles=legend_handles(plt), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_effects(plt, effects, path):
    """Forest plot of paired differences, one panel per metric."""
    from matplotlib.ticker import MaxNLocator

    keys = [k for k in FIG_METRICS if any(e["res"].get(k) for e in effects)]
    better = {k: hb for k, _, hb in METRICS}
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 0.45 * len(effects) * nrow + 1.4), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(effects))[::-1]
    for ax, k in zip(axes, keys):
        ax.axvline(0, color=MUTED, linewidth=1)
        for i in range(1, len(effects)):  # separate groups of comparisons (fine-tuning / RAG / ...)
            if effects[i]["label"].split(" · ")[0] != effects[i - 1]["label"].split(" · ")[0]:
                ax.axhline(len(effects) - i - 0.5, color=GRID, linewidth=1)
        for yi, e in zip(y, effects):
            r = e["res"].get(k)
            if not r:
                continue
            color = C_YANDEX if family(e["a"]) == "yandex" else C_QWEN
            if e["a"].startswith("yandex") and e["b"].startswith("qwen"):
                color = INK2  # cross-model comparison
            ax.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(100 * r["diff"], yi, marker="o", markersize=7, color=color, markeredgewidth=1.6,
                    markerfacecolor=color if r["sig"] else SURFACE, linestyle="none", zorder=3)
            if r["sig"]:
                ax.annotate(f"{100 * r['diff']:+.0f}", (100 * r["hi"], yi), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK)
        ax.set_title(SHORT[k], fontsize=9.5)
        ax.set_xlabel("разница, п.п.   " + ("лучше →" if better[k] else "← лучше"), fontsize=8, color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=False))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
        ax.grid(axis="y", visible=False)
        lim = max(10, *(abs(100 * e["res"][k][b]) for e in effects if e["res"].get(k) for b in ("lo", "hi")))
        ax.set_xlim(-lim * 1.15, lim * 1.25)
    for ax in axes[len(keys):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_yticks(y)
        ax.set_yticklabels([e["label"] for e in effects])
    fig.suptitle("Эффекты вмешательств: парная разница A − B по тем же запросам, 95% ДИ. "
                 "Закрашено — интервал не пересекает 0", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_dishes(plt, rows_by_sys, systems, path, max_rows=25):
    """Which classic dishes each system gets wrong: only dishes with at least one miss are drawn,
    colour = share of key ingredients MISSED (so the eye goes to errors), cell text = found / total."""
    dish_ids = sorted({pid for s in systems for pid, r in rows_by_sys[s].items() if r.get("type") == "dish"})
    if not dish_ids:
        return None
    ref = {pid: next(rows_by_sys[s][pid] for s in systems if pid in rows_by_sys[s]) for pid in dish_ids}
    R = np.array([[rows_by_sys[s].get(pid, {}).get("metrics", {}).get("must_have_recall", np.nan) for s in systems]
                  for pid in dish_ids])
    bad = np.nansum(1 - R, 1) > 1e-9
    if not bad.any():
        return {"shown": 0, "total": len(dish_ids)}
    R = R[bad]
    dish_ids = [p for p, b in zip(dish_ids, bad) if b]
    n_bad = len(dish_ids)
    order = np.argsort(-np.nansum(1 - R, 1), kind="stable")[:max_rows]  # most-missed dishes on top
    R = R[order]
    dish_ids = [dish_ids[i] for i in order]
    miss = 1 - R
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f1f6fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(0.95 * len(systems) + 3.0, 0.36 * len(dish_ids) + 1.9))
    ax.imshow(miss, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=30, ha="left")
    ax.set_yticks(range(len(dish_ids)))
    ax.set_yticklabels([ref[p].get("dish", p) for p in dish_ids])
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(-.5, len(systems)), minor=True)
    ax.set_yticks(np.arange(-.5, len(dish_ids)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    fam = [family(s) for s in systems]
    for j in range(1, len(systems)):
        if fam[j] != fam[j - 1]:
            ax.axvline(j - 0.5, color=INK2, linewidth=1.2)
    for i, pid in enumerate(dish_ids):
        n = len(ref[pid].get("must_have") or []) or 1
        for j in range(len(systems)):
            v = R[i, j]
            if np.isnan(v):
                continue
            found = int(round(v * n))
            ax.text(j, i, "✓" if found == n else f"{found} из {n}", ha="center", va="center",
                    fontsize=8 if found < n else 9, color=(SURFACE if miss[i, j] > 0.6 else INK) if found < n else MUTED)
    head = ("Классические блюда, где хотя бы одна система пропустила ключевой ингредиент" if n_bad <= max_rows else
            f"{len(dish_ids)} блюд с наибольшим числом пропусков ключевых ингредиентов (всего с пропусками: {n_bad})")
    ax.set_title(head + "\n"
                 "в ячейке — сколько ключевых ингредиентов найдено; чем темнее, тем больше пропущено",
                 fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"shown": len(dish_ids), "with_misses": n_bad}


def training_runs(patterns):
    try:
        import mlflow
    except ImportError:
        return {}
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
    out = {}
    try:
        c = mlflow.MlflowClient()
        for label, pat in patterns.items():
            df = mlflow.search_runs(experiment_names=["recipe-llm"], filter_string=f"tags.mlflow.runName LIKE '{pat}'",
                                    order_by=["attributes.start_time DESC"])
            for _, r in df.iterrows():
                h = sorted({m.step: m.value for m in c.get_metric_history(r.run_id, "eval_loss")}.items())
                if len(h) >= 3:
                    out[label] = {"run": r["tags.mlflow.runName"], "eval_loss": h,
                                  "train_loss": sorted((m.step, m.value) for m in c.get_metric_history(r.run_id, "loss"))}
                    break
    except Exception as e:  # report still builds without MLflow
        print(f"[warn] MLflow: {e}")
    return out


def fig_training(plt, runs, path):
    """One panel per model (different tokenizers -> losses are not comparable on one axis)."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (label, r) in zip(axes, runs.items()):
        color = C_YANDEX if "yandex" in label.lower() else C_QWEN
        if r["train_loss"]:
            s, v = zip(*r["train_loss"])
            k = max(1, len(v) // 40)
            sm = np.convolve(v, np.ones(k) / k, mode="valid")
            ax.plot(s[k - 1:], sm, color=color, alpha=0.35, linewidth=1.5, label="train loss (скольз. среднее)")
        s, v = zip(*r["eval_loss"])
        ax.plot(s, v, color=color, linewidth=2, marker="o", markersize=5, label="eval loss")
        best = int(np.argmin(v))
        ax.annotate(f"лучший: {v[best]:.3f}\n(шаг {s[best]})", (s[best], v[best]), xytext=(-10, 18),
                    textcoords="offset points", ha="right", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
        ax.set_title(f"{label}  ·  {r['run']}")
        ax.set_xlabel("шаг оптимизатора")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Обучение LoRA: потери по шагам (у моделей разные токенизаторы — абсолютные значения не сравнимы)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_judge(plt, path):
    p = os.path.join(ROOT, "data", "clean", "scored.parquet")
    if not os.path.exists(p):
        return None
    import pandas as pd

    s = pd.read_parquet(p, columns=["judge_score"])["judge_score"].to_numpy()
    t = pd.read_parquet(os.path.join(ROOT, "data", "clean", "train.parquet"), columns=["judge_score"])["judge_score"]
    cut = float(t.min())
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.linspace(1, 5, 41)
    ax.hist(s[s < cut], bins=bins, color="#9ec5f4", edgecolor=SURFACE, linewidth=1, label="отброшены")
    ax.hist(s[s >= cut], bins=bins, color=C_YANDEX, edgecolor=SURFACE, linewidth=1, label=f"в обучение (≥ {cut:.2f})")
    ax.axvline(cut, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("оценка LLM-судьи (матожидание по цифрам 1–5)")
    ax.set_ylabel("рецептов")
    ax.set_title(f"LLM-as-judge: {len(s):,} кандидатов → {len(t):,} в обучающую выборку".replace(",", " "))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"scored": int(len(s)), "selected": int(len(t)), "cut": cut}


def _ranks(x):
    import pandas as pd

    return pd.Series(x).rank(method="average").to_numpy()


def _auc(score, pos):
    """P(score of a random positive > score of a random negative), ties count 1/2 (Mann-Whitney)."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return np.nan
    return (_ranks(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def judge_validation(rng, B):
    """Agreement of the LLM judge with blind reference labels (scripts/make_judge_labeling.py).
    The labels file may name its annotator: {"annotator": "человек" | "DeepSeek-R1" | ..., "items": [...]}."""
    lp = os.path.join(ROOT, "eval", "judge_labels.json")
    sp = os.path.join(ROOT, "data", "clean", "scored.parquet")
    if not (os.path.exists(lp) and os.path.exists(sp)):
        return None
    import pandas as pd

    with open(lp, encoding="utf-8") as f:
        raw = json.load(f)
    lab = pd.DataFrame(raw["items"])
    sc = pd.read_parquet(sp, columns=["recipe_id", "judge_score"]).dropna()
    sc["recipe_id"] = sc["recipe_id"].astype(int)
    df = lab.merge(sc, on="recipe_id", how="inner")
    if len(df) < 10:
        return None
    cut = float(pd.read_parquet(os.path.join(ROOT, "data", "clean", "train.parquet"), columns=["judge_score"])
                ["judge_score"].min())
    j, h = df["judge_score"].to_numpy(float), df["human"].to_numpy(float)

    def stats(idx):
        jj, hh = j[idx], h[idx]
        sel = jj >= cut
        rho = np.corrcoef(_ranks(jj), _ranks(hh))[0, 1] if jj.std() and hh.std() else np.nan
        return {"spearman": rho, "auc_good": _auc(jj, hh >= 4), "auc_bad": _auc(-jj, hh <= 2),
                "bad_selected": (hh[sel] <= 2).mean() if sel.any() else np.nan,
                "bad_dropped": (hh[~sel] <= 2).mean() if (~sel).any() else np.nan,
                "good_selected": (hh[sel] >= 4).mean() if sel.any() else np.nan,
                "good_dropped": (hh[~sel] >= 4).mean() if (~sel).any() else np.nan}

    point = stats(np.arange(len(df)))
    boots = [stats(rng.integers(0, len(df), len(df))) for _ in range(min(B, 4000))]
    out = {"n": len(df), "cut": cut, "n_selected": int((j >= cut).sum()), "ref": raw.get("annotator") or "эталон"}
    for k, v in point.items():
        arr = np.array([b[k] for b in boots], float)
        arr = arr[~np.isnan(arr)]
        out[k] = (float(v), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (float(v), np.nan, np.nan)
    out["j"], out["h"] = j.tolist(), h.tolist()
    return out


def fig_judge_validation(plt, jv, path):
    j, h, cut, ref = np.array(jv["j"]), np.array(jv["h"]), jv["cut"], jv["ref"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    jit = np.random.default_rng(0).uniform(-0.17, 0.17, len(h))
    ax.scatter(j, h + jit, s=16, color=C_YANDEX, alpha=0.45, linewidths=0)
    edges = np.quantile(j, np.linspace(0, 1, 6))
    xs, ms, los, his = [], [], [], []
    rng = np.random.default_rng(1)
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (j >= lo_e) & (j <= hi_e)
        if m.sum() < 3:
            continue
        mean, lo, hi = boot_mean(h[m], rng, 2000)
        xs.append(j[m].mean()); ms.append(mean); los.append(lo); his.append(hi)
    ax.vlines(xs, los, his, color=INK, linewidth=2)
    ax.plot(xs, ms, color=INK, linewidth=1.5, marker="o", markersize=6, label="средняя оценка разметки (квинтили)")
    ax.axvline(cut, color=INK2, linewidth=1, linestyle="--")
    ax.annotate(f"порог отбора {cut:.2f}", (cut, 5.35), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=INK2, va="center")
    ax.set_ylim(0.6, 5.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("оценка LLM-судьи")
    ax.set_ylabel(f"оценка: {ref}")
    r = jv["spearman"]
    ax.set_title(f"Судья vs {ref}: ρ = {r[0]:.2f} [{r[1]:.2f}; {r[2]:.2f}], n = {jv['n']}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    groups = [("отобраны", h[j >= cut]), ("отброшены", h[j < cut])]
    for yi, (name, hh) in enumerate(groups):
        left = 0.0
        for k in range(1, 6):
            share = float((hh == k).mean()) if len(hh) else 0.0
            bx.barh(yi, share, left=left, color=ramp[k - 1], edgecolor=SURFACE, linewidth=2, height=0.6)
            if share >= 0.08:
                bx.text(left + share / 2, yi, f"{100 * share:.0f}%", ha="center", va="center", fontsize=8,
                        color=SURFACE if k >= 3 else INK)
            left += share
    bx.set_yticks([0, 1])
    bx.set_yticklabels([f"{n}\n(n = {len(hh)})" for n, hh in groups])
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    bx.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    bx.legend(handles=[Patch(color=c, label=str(k)) for k, c in enumerate(ramp, 1)], title=f"оценка: {ref}",
              ncol=5, frameon=False, fontsize=8, title_fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    bx.set_title("Оценки разметки: отобранные и отброшенные")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="*", default=["YandexGPT-5-Lite-8B=ty-%", "Qwen2.5-7B=tq-%"],
                    help="label=MLflow run name pattern for training curves")
    a = ap.parse_args()

    found = {os.path.basename(p)[:-len(".jsonl")] for p in glob.glob(os.path.join(RES, "*.jsonl"))}
    systems = a.systems or [s for s in DEFAULT_ORDER if s in found] + sorted(found - set(DEFAULT_ORDER))
    if not systems:
        raise SystemExit(f"No results in {RES}")
    rows = {s: load(s) for s in systems}
    rng = np.random.default_rng(a.seed)
    B = a.bootstrap

    table = {s: {} for s in systems}
    for s in systems:
        for k, _, _ in METRICS:
            v = values(rows[s], k)
            if v:
                table[s][k] = boot_mean(np.array(list(v.values())), rng, B)

    effects = []
    for label, sa, sb in EFFECTS:
        if sa in rows and sb in rows:
            res = {k: boot_paired(values(rows[sa], k), values(rows[sb], k), rng, B) for k, _, _ in METRICS}
            effects.append({"label": label, "a": sa, "b": sb, "res": {k: v for k, v in res.items() if v}})

    os.makedirs(FIG, exist_ok=True)
    plt = setup_mpl()
    figs = {}
    from collections import Counter

    types = Counter(r.get("type") for r in rows[systems[0]].values())
    n_prompts = sum(types.values())
    fig_metrics(plt, table, systems, os.path.join(FIG, "metrics_ci.png"), n_prompts)
    figs["metrics"] = "figures/metrics_ci.png"
    if effects:
        fig_effects(plt, effects, os.path.join(FIG, "effects.png")); figs["effects"] = "figures/effects.png"
    dishes = fig_dishes(plt, rows, systems, os.path.join(FIG, "dishes_heatmap.png"))
    if dishes and dishes["shown"]:
        figs["dishes"] = "figures/dishes_heatmap.png"
    runs = training_runs(dict(x.split("=", 1) for x in a.runs))
    if runs:
        fig_training(plt, runs, os.path.join(FIG, "training.png")); figs["training"] = "figures/training.png"
    judge = fig_judge(plt, os.path.join(FIG, "judge_scores.png"))
    if judge:
        figs["judge"] = "figures/judge_scores.png"
    jv = judge_validation(rng, B)
    if jv:
        fig_judge_validation(plt, jv, os.path.join(FIG, "judge_validation.png"))
        figs["judge_validation"] = "figures/judge_validation.png"

    # ---- markdown
    L = ["# Отчёт об оценке", "",
         f"Набор: `eval/prompts.jsonl` — {n_prompts} запросов ({types['dish']} классических блюд, "
         f"{types['ingredients']} наборов продуктов, {types['category']} категорий), жадное декодирование. "
         f"Интервалы — 95% перцентильный бутстрэп по запросам, B = {B}. Метрика «ключевые ингредиенты» "
         f"считается по {types['dish']} блюдам, «продукты из запроса» — по {types['ingredients']} наборам, "
         f"остальные — по всем {n_prompts} запросам.", ""]
    L += ["## Метрики по системам", "",
          "| Система | " + " | ".join(f"{l} {'↑' if hb else '↓'}" for _, l, hb in METRICS) + " |",
          "|---|" + "---|" * len(METRICS)]
    for s in systems:
        cells = []
        for k, _, _ in METRICS:
            st = table[s].get(k)
            cells.append(f"{pct(st[0])} <sub>[{pct(st[1])}–{pct(st[2])}]</sub>" if st else "—")
        L.append(f"| `{s}` | " + " | ".join(cells) + " |")
    L += ["", f"![Метрики с доверительными интервалами]({figs['metrics']})", ""]

    if effects:
        L += ["## Эффекты: парные разницы", "",
              "Разница A − B в процентных пунктах по одним и тем же запросам. ✅/⚠️ — интервал не пересекает 0 "
              "(✅ — в лучшую сторону, ⚠️ — в худшую); пусто — эффект не отличим от шума.", "",
              "| Сравнение (A − B) | " + " | ".join(l for _, l, _ in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
        for e in effects:
            cells = []
            for k, _, hb in METRICS:
                r = e["res"].get(k)
                if not r:
                    cells.append("—")
                    continue
                mark = ("✅ " if (r["diff"] > 0) == hb else "⚠️ ") if r["sig"] else ""
                cells.append(f"{mark}{100 * r['diff']:+.0f} <sub>[{100 * r['lo']:+.0f}; {100 * r['hi']:+.0f}]</sub>")
            L.append(f"| {e['label']}<br><sub>`{e['a']}` − `{e['b']}`</sub> | " + " | ".join(cells) + " |")
        L += ["", f"![Эффекты]({figs['effects']})", ""]

    if "dishes" in figs:
        n_dish = len({p for s in systems for p, r in rows[s].items() if r.get("type") == "dish"})
        L += ["## Где ошибаются модели: классические блюда", "",
              f"Хотя бы одна система пропустила ключевой ингредиент в {dishes['with_misses']} из {n_dish} блюд; "
              f"на рисунке — {dishes['shown']} блюд с наибольшим числом пропусков.", "", f"![Блюда]({figs['dishes']})", ""]
    if "training" in figs:
        L += ["## Обучение", ""]
        for label, r in runs.items():
            v = [x[1] for x in r["eval_loss"]]
            s = [x[0] for x in r["eval_loss"]]
            L.append(f"- **{label}** (`{r['run']}`): eval loss {v[0]:.3f} → {min(v):.3f} "
                     f"(лучший на шаге {s[int(np.argmin(v))]} из {s[-1]}); после шага {s[1]}: {v[1]:.3f}.")
        L += ["", f"![Обучение]({figs['training']})", ""]
    if judge:
        L += ["## Данные: отбор LLM-судьёй", "",
              f"{judge['scored']:,} рецептов-кандидатов оценены базовой YandexGPT; в обучение вошли "
              f"{judge['selected']:,} лучших (оценка ≥ {judge['cut']:.2f}).".replace(",", " "), "",
              f"![Оценки судьи]({figs['judge']})", ""]
    if jv:
        ci = lambda k, f=lambda x: f"{x:.2f}": f"{f(jv[k][0])} [{f(jv[k][1])}; {f(jv[k][2])}]"
        pc = lambda x: f"{100 * x:.0f}%"
        ref = jv["ref"]
        L += [f"## Проверка судьи независимой разметкой: {ref}", "",
              f"{jv['n']} рецептов, стратифицированных по оценке судьи (поровну из каждого дециля, поэтому выборка "
              f"репрезентативна), оценены вслепую по шкале 1–5; разметчик: {ref}. В скобках — 95% бутстрэп-интервал.", "",
              "| Показатель | Значение |", "|---|---|",
              f"| ρ Спирмена (судья vs {ref}) | {ci('spearman')} |",
              f"| AUC: судья отличает хорошие (≥ 4 по разметке) от остальных | {ci('auc_good')} |",
              f"| AUC: судья отличает плохие (≤ 2 по разметке) от остальных | {ci('auc_bad')} |",
              f"| Доля плохих (≤ 2) среди отобранных / отброшенных | {ci('bad_selected', pc)} / {ci('bad_dropped', pc)} |",
              f"| Доля хороших (≥ 4) среди отобранных / отброшенных | {ci('good_selected', pc)} / {ci('good_dropped', pc)} |",
              "", f"![Судья vs {ref}]({figs['judge_validation']})", ""]
    sbs = sorted(glob.glob(os.path.join(ROOT, "eval", "sbs_*_results.json")))
    if sbs:
        L += ["## Слепое сравнение (SBS)", "", "| A | B | Побед A | Ничьих | Побед B | Win rate B [95% ДИ] |",
              "|---|---|---|---|---|---|"]
        for p in sbs:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            w = np.array([{"a": 0, "tie": 0.5, "b": 1}[x["winner"]] for x in r["items"]])
            m, lo, hi = boot_mean(w, rng, B) if len(w) else (0, 0, 0)
            L.append(f"| `{r['a']}` | `{r['b']}` | {r['a_wins']} | {r['ties']} | {r['b_wins']} | "
                     f"{pct(m)} [{pct(lo)}–{pct(hi)}] |")
        L.append("")

    out = os.path.join(ROOT, "eval", "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(ROOT, "eval", "report_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"table": table, "effects": effects,
                   "judge_validation": {k: v for k, v in (jv or {}).items() if k not in ("j", "h")}},
                  f, ensure_ascii=False, indent=1)
    print(f"-> {out}\n-> {FIG}/ ({', '.join(os.path.basename(v) for v in figs.values())})")


if __name__ == "__main__":
    main()
