#!/usr/bin/env python
"""Blind human labeling page to validate the LLM judge.

  python scripts/make_judge_labeling.py            # -> eval/judge_labeling.html (120 recipes)

The sample is stratified by the judge score (equal number of recipes from each score decile), so it is
representative of all candidates. The page shows each recipe exactly as the judge saw it, with the same
1-5 rubric, and does NOT show the judge score. Label with keys 1..5 (about 40 min for 120 recipes; progress
is kept in the browser), download judge_labels.json at the end, put it into eval/ and run
scripts/report.py: it adds agreement statistics (Spearman, AUC, share of bad recipes among selected vs
dropped) with bootstrap CIs.

The labels may also come from a stronger model instead of a person: then write the same JSON with
{"annotator": "<model name>", "items": [{"recipe_id": ..., "human": 1..5}, ...]} and the report names it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from recipe_llm.prompts import format_recipe  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка рецептов</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e3e1dc;--accent:#2a78d6;--sel:#eaf2fc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21201e;--ink:#ecebe8;--muted:#9a9893;--line:#34322f;--accent:#5b9ae6;--sel:#1d2a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
.wrap{max-width:860px;margin:0 auto}.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}.bar{height:4px;background:var(--line);border-radius:2px;margin-top:10px}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:2px;transition:width .2s}
details{margin:12px 16px;font-size:13px}summary{cursor:pointer;color:var(--muted)}details ol{margin:6px 0;padding-left:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:12px 16px 120px;min-width:0}
.md{overflow-wrap:anywhere}.md h4{margin:4px 0 8px;font-size:18px}.md h5{margin:12px 0 4px;font-size:14px;color:var(--muted)}
.md p{margin:4px 0}.md li{margin:2px 0}.md ul,.md ol{margin:4px 0;padding-left:22px}
.actions{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:6px;justify-content:center;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 12px;border-radius:8px;cursor:pointer;min-width:44px}
button:hover{border-color:var(--accent)}button.on{border-color:var(--accent);background:var(--sel)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:6px;color:var(--muted)}
.result{max-width:640px;margin:40px auto;padding:0 16px}
</style></head><body>
<header><div class="wrap"><div class="top"><h1>Оцените рецепт по шкале 1–5</h1><span class="muted" id="count"></span></div><div class="bar"><i id="bar"></i></div></div></header>
<main class="wrap"><details open><summary>Шкала (та же, что у LLM-судьи)</summary><ol>
<li>бессмыслица, ошибки, блюдо так не приготовить;</li><li>серьёзные проблемы: пропущены важные шаги или ингредиенты, странные количества;</li>
<li>приготовить можно, но есть заметные недочёты;</li><li>хороший понятный рецепт с мелкими недочётами;</li>
<li>отличный рецепт: соответствует названию, логичные шаги, разумные количества.</li></ol></details>
<div id="main"></div></main>
<div class="actions" id="actions"></div>
<script>
const DATA = __DATA__;
const KEY = "judge-labels:" + DATA.tag;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = DATA.items.findIndex(x => !(x.id in labels)); if (i < 0) i = DATA.items.length;
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function esc(s){ return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function md(t){
  const out=[]; let list=null;
  const close=()=>{ if(list){ out.push("</"+list+">"); list=null; } };
  for (const raw of t.split("\n")) {
    const l = raw.trim(); if (!l) { close(); continue; }
    let m;
    if ((m = l.match(/^#\s+(.*)/))) { close(); out.push("<h4>"+esc(m[1])+"</h4>"); }
    else if ((m = l.match(/^#{2,6}\s+(.*)/))) { close(); out.push("<h5>"+esc(m[1])+"</h5>"); }
    else if ((m = l.match(/^[-*•]\s+(.*)/))) { if(list!=="ul"){ close(); out.push("<ul>"); list="ul"; } out.push("<li>"+esc(m[1])+"</li>"); }
    else if ((m = l.match(/^\d+[.)]\s+(.*)/))) { if(list!=="ol"){ close(); out.push("<ol>"); list="ol"; } out.push("<li>"+esc(m[1])+"</li>"); }
    else { close(); out.push("<p>"+esc(l).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>"); }
  }
  close(); return out.join("");
}
function buttons(){
  const v = i < DATA.items.length ? labels[DATA.items[i].id] : null;
  let h = "";
  for (let k = 1; k <= 5; k++) h += '<button class="'+(v===k?"on":"")+'" onclick="vote('+k+')">'+k+'<kbd>'+k+'</kbd></button>';
  h += '<button onclick="back()">Назад<kbd>⌫</kbd></button>';
  document.getElementById("actions").innerHTML = h;
}
function render(){
  const n = DATA.items.length;
  document.getElementById("count").textContent = Math.min(i+1, n) + " / " + n;
  document.getElementById("bar").style.width = (100*Object.keys(labels).length/n) + "%";
  if (i >= n) return finish();
  document.getElementById("main").innerHTML = '<div class="card"><div class="md">'+md(DATA.items[i].text)+'</div></div>';
  buttons(); window.scrollTo(0,0);
}
function vote(k){ if (i >= DATA.items.length) return; labels[DATA.items[i].id] = k; save(); i++; render(); }
function back(){ if (i > 0) { i--; render(); } }
function finish(){
  document.getElementById("actions").style.display = "none";
  const items = DATA.items.filter(x => x.id in labels).map(x => ({recipe_id: x.id, human: labels[x.id]}));
  const res = {tag: DATA.tag, annotator: "человек", n: items.length, items};
  const blob = URL.createObjectURL(new Blob([JSON.stringify(res, null, 1)], {type:"application/json"}));
  document.getElementById("main").innerHTML = '<div class="result"><h2>Готово: '+items.length+' рецептов</h2>'+
    '<p><a download="judge_labels.json" href="'+blob+'"><button class="primary">Скачать judge_labels.json</button></a> '+
    '<button onclick="i=0;document.getElementById(\'actions\').style.display=\'flex\';render()">Просмотреть ещё раз</button></p>'+
    '<p class="muted">Положите файл в папку eval/ проекта и запустите scripts/report.py.</p></div>';
}
document.addEventListener("keydown", e => {
  if (e.key >= "1" && e.key <= "5") vote(+e.key);
  else if (e.key==="Backspace") { e.preventDefault(); back(); }
});
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default=os.path.join(ROOT, "data/clean/scored.parquet"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_recipe_chars", type=int, default=3500, help="same truncation as score_recipes.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/judge_labeling.html"))
    a = ap.parse_args()

    df = pd.read_parquet(a.scored)
    df = df[df["recipe_id"].notna()].copy()
    # Equal-count bins over the judge score: each bin is the same share of the population, so drawing the same
    # number from every bin keeps the sample representative while guaranteeing coverage of low and high scores.
    df["bin"] = pd.qcut(df["judge_score"].rank(method="first"), a.bins, labels=False)
    per_bin = int(np.ceil(a.n / a.bins))
    sample = df.groupby("bin", group_keys=False).apply(lambda g: g.sample(min(per_bin, len(g)), random_state=a.seed))
    sample = sample.sample(frac=1, random_state=a.seed).head(a.n)  # shuffle: score order must not leak

    items = [{"id": int(r["recipe_id"]), "text": format_recipe(r, include_description=True)[:a.max_recipe_chars]}
             for r in sample.to_dict("records")]
    data = json.dumps({"tag": f"n{a.n}-s{a.seed}", "items": items}, ensure_ascii=False).replace("</", "<\\/")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", data))
    print(f"{len(items)} recipes (stratified over {a.bins} judge-score bins) -> {a.out}\n"
          "Open it in a browser, label with keys 1-5, download judge_labels.json into eval/, run scripts/report.py")


if __name__ == "__main__":
    main()
