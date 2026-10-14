#!/usr/bin/env python
"""Build a blind side-by-side (SBS) comparison page for two evaluated systems.

  python scripts/make_sbs.py yandex-base yandex-ft      # -> eval/sbs_yandex-base_vs_yandex-ft.html
  python scripts/make_sbs.py yandex-base yandex-ft --n 0   # all prompts (default: random 60, ~20 min of labeling)

Open the HTML file in any browser (no server needed): for each request you see two
recipes in random left/right order and pick the better one (keys 1 / 2 / 3). At the end
the page reveals the result and lets you download eval/sbs_<a>_vs_<b>_results.json;
put it into eval/ and run scripts/report.py (bootstrap CI on the win rate) or scripts/compare.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SBS: рецепты</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e3e1dc;--accent:#e5482d;--sel:#fff1ec}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21201e;--ink:#ecebe8;--muted:#9a9893;--line:#34322f;--accent:#ff6a4d;--sel:#3a2621}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
.wrap{max-width:1200px;margin:0 auto}.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}.bar{height:4px;background:var(--line);border-radius:2px;margin-top:10px}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:2px;transition:width .2s}
.prompt{font-size:18px;font-weight:600;margin:18px 16px 4px}.hint{margin:0 16px 12px;font-size:13px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 16px}@media (max-width:760px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.md{overflow-wrap:anywhere}.card.chosen{border-color:var(--accent);background:var(--sel)}.card h3{margin:0 0 8px;font-size:13px;color:var(--muted);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.md h4{margin:10px 0 4px;font-size:15px}.md h5{margin:10px 0 4px;font-size:14px;color:var(--muted)}.md p{margin:4px 0}.md li{margin:2px 0}.md ul,.md ol{margin:4px 0;padding-left:22px}
.actions{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:9px 16px;border-radius:8px;cursor:pointer}
button:hover{border-color:var(--accent)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:6px;color:var(--muted)}
.result{max-width:640px;margin:40px auto;padding:0 16px}.result table{border-collapse:collapse;width:100%;margin:12px 0}.result td{border-bottom:1px solid var(--line);padding:6px 4px}
</style></head><body>
<header><div class="wrap"><div class="top"><h1>Слепое сравнение рецептов</h1><span class="muted" id="count"></span></div><div class="bar"><i id="bar"></i></div></div></header>
<main class="wrap" id="main"></main>
<div class="actions" id="actions">
<button onclick="vote('L')">Левый лучше<kbd>1</kbd></button><button onclick="vote('T')">Одинаково<kbd>2</kbd></button>
<button onclick="vote('R')">Правый лучше<kbd>3</kbd></button><button onclick="back()">Назад<kbd>⌫</kbd></button></div>
<script>
const DATA = __DATA__;
const KEY = "sbs:" + DATA.a + ":" + DATA.b;
let votes = {};
try { votes = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = DATA.items.findIndex(x => !(x.id in votes)); if (i < 0) i = DATA.items.length;
function save(){ try { localStorage.setItem(KEY, JSON.stringify(votes)); } catch (e) {} }
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
function render(){
  const n = DATA.items.length;
  document.getElementById("count").textContent = Math.min(i+1, n) + " / " + n;
  document.getElementById("bar").style.width = (100*Object.keys(votes).length/n) + "%";
  if (i >= n) return finish();
  const it = DATA.items[i], v = votes[it.id];
  document.getElementById("main").innerHTML =
    '<div class="prompt">'+esc(it.prompt)+'</div><p class="hint muted">Какой рецепт лучше: правильнее, понятнее, реально приготовить?</p>'+
    '<div class="cols"><div class="card'+(v==="L"?" chosen":"")+'"><h3>Левый</h3><div class="md">'+md(it.left)+'</div></div>'+
    '<div class="card'+(v==="R"?" chosen":"")+'"><h3>Правый</h3><div class="md">'+md(it.right)+'</div></div></div>';
  window.scrollTo(0,0);
}
function vote(c){ if (i >= DATA.items.length) return; votes[DATA.items[i].id] = c; save(); i++; render(); }
function back(){ if (i > 0) { i--; render(); } }
function finish(){
  document.getElementById("actions").style.display = "none";
  let a=0,b=0,t=0; const items=[];
  for (const it of DATA.items) {
    const v = votes[it.id]; if (!v) continue;
    const w = v==="T" ? "tie" : ((v==="L") === it.a_left ? "a" : "b");
    if (w==="a") a++; else if (w==="b") b++; else t++;
    items.push({id: it.id, prompt: it.prompt, winner: w});
  }
  const n=a+b+t, wr = n ? (b+0.5*t)/n : 0;
  const res = {a: DATA.a, b: DATA.b, a_wins: a, ties: t, b_wins: b, win_rate_b: wr, items};
  const blob = URL.createObjectURL(new Blob([JSON.stringify(res, null, 2)], {type:"application/json"}));
  document.getElementById("main").innerHTML = '<div class="result"><h2>Готово</h2><table>'+
    '<tr><td>'+esc(DATA.a)+' лучше</td><td><b>'+a+'</b></td></tr><tr><td>Одинаково</td><td><b>'+t+'</b></td></tr>'+
    '<tr><td>'+esc(DATA.b)+' лучше</td><td><b>'+b+'</b></td></tr><tr><td>Win rate '+esc(DATA.b)+'</td><td><b>'+Math.round(100*wr)+'%</b></td></tr></table>'+
    '<p><a download="sbs_'+DATA.a+'_vs_'+DATA.b+'_results.json" href="'+blob+'"><button class="primary">Скачать результаты</button></a> '+
    '<button onclick="votes={};save();location.reload()">Начать заново</button></p>'+
    '<p class="muted">Положите файл в папку eval/ проекта и запустите scripts/report.py.</p></div>';
}
document.addEventListener("keydown", e => {
  if (e.key==="1"||e.key==="ArrowLeft") vote("L"); else if (e.key==="2"||e.key==="ArrowDown") vote("T");
  else if (e.key==="3"||e.key==="ArrowRight") vote("R"); else if (e.key==="Backspace") { e.preventDefault(); back(); }
});
render();
</script></body></html>
"""


def load(name):
    with open(os.path.join(ROOT, "eval/results", f"{name}.jsonl"), encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=60, help="random subset of prompts to compare, 0 = all")
    args = ap.parse_args()
    a, b = args.a, args.b
    ra, rb = load(a), load(b)
    rng = random.Random(f"{a}|{b}")
    ids = sorted(k for k in ra if k in rb)
    if args.n and len(ids) > args.n:  # stratified by prompt type, same subset on every run
        by_type = {}
        for k in ids:
            by_type.setdefault(ra[k].get("type"), []).append(k)
        sub = []
        for t, ks in sorted(by_type.items(), key=lambda x: str(x[0])):
            sub += random.Random(f"{a}|{b}|{t}").sample(ks, max(1, round(args.n * len(ks) / len(ids))))
        ids = sub
    items = []
    for pid in ids:
        a_left = rng.random() < 0.5
        x, y = ra[pid]["response"], rb[pid]["response"]
        items.append({"id": pid, "prompt": ra[pid]["prompt"], "a_left": a_left,
                      "left": x if a_left else y, "right": y if a_left else x})
    rng.shuffle(items)
    data = json.dumps({"a": a, "b": b, "items": items}, ensure_ascii=False).replace("</", "<\\/")
    out = os.path.join(ROOT, "eval", f"sbs_{a}_vs_{b}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", data))
    print(f"{len(items)} pairs -> {out}")


if __name__ == "__main__":
    main()
