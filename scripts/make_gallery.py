#!/usr/bin/env python
"""Static gallery of model answers for GitHub Pages: every eval prompt, answers of all systems side by side.

  python scripts/make_gallery.py          # eval/results/*.jsonl -> docs/index.html (self-contained, no server)

Publish: commit docs/, then on GitHub: Settings -> Pages -> Source "Deploy from a branch", branch main, folder /docs.
Only model answers, prompts, metric flags and titles of retrieved recipes go into the page; full texts of
retrieved povarenok.ru recipes are left out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["yandex-base", "yandex-base-rag", "yandex-ft", "yandex-ft-rag",
         "qwen-base", "qwen-base-rag", "qwen-ft", "qwen-ft-rag"]
MODEL = {"yandex": "YandexGPT-5 Lite 8B", "qwen": "Qwen2.5 7B"}


def label(name: str) -> str:
    fam = "yandex" if name.startswith("yandex") else "qwen" if name.startswith("qwen") else ""
    if not fam:
        return name
    variant = "LoRA" if "-ft" in name else "исходная"
    return f"{MODEL[fam]} · {variant}" + (" + RAG" if name.endswith("-rag") else "")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def flags(r: dict) -> dict:
    m = r.get("metrics", {})
    out = {"chars": len(r.get("response") or "")}
    if r.get("must_have") and "must_have_recall" in m:
        n = len(r["must_have"])
        out["key"] = [round(m["must_have_recall"] * n), n]
    if r.get("requested") and "requested_coverage" in m:
        n = len([q for q in r["requested"] if q.strip()])
        out["req"] = [round(m["requested_coverage"] * n), n]
    for k in ("truncated", "chatter"):
        if m.get(k):
            out[k] = 1
    if "structure_ok" in m and not m["structure_ok"]:
        out["nostruct"] = 1
    if r.get("retrieved"):
        out["rag"] = [x.get("title", "") for x in r["retrieved"]][:3]
    return out


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Галерея рецептов</title>
<meta name="description" content="Ответы 8 систем (YandexGPT-5 Lite и Qwen2.5, с LoRA и RAG) на 200 запросов про рецепты">
<style>
:root{--bg:#f7f6f3;--surface:#fff;--ink:#1c1b19;--ink2:#52514e;--muted:#8a8984;--line:#e6e4df;--hover:#f0eee9;
--sel:#e9f1fb;--accent:#2a78d6;--yandex:#2a78d6;--qwen:#eb6834;--warn-bg:#fdf0e6;--warn:#9a4a12;--ok-bg:#eaf4ec;--ok:#2e6b3a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--surface:#1f1e1c;--ink:#ecebe7;--ink2:#b9b7b1;
--muted:#8d8b85;--line:#32302d;--hover:#292826;--sel:#1b2a3d;--accent:#5b9ae6;--yandex:#5b9ae6;--qwen:#f08a5d;
--warn-bg:#3a2617;--warn:#f3b48a;--ok-bg:#1c3022;--ok:#9fd3a9}}
:root[data-theme="dark"]{--bg:#151514;--surface:#1f1e1c;--ink:#ecebe7;--ink2:#b9b7b1;--muted:#8d8b85;--line:#32302d;
--hover:#292826;--sel:#1b2a3d;--accent:#5b9ae6;--yandex:#5b9ae6;--qwen:#f08a5d;--warn-bg:#3a2617;--warn:#f3b48a;--ok-bg:#1c3022;--ok:#9fd3a9}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
a{color:var(--accent)}button{font:inherit;color:inherit}
header{padding:20px 24px 14px;border-bottom:1px solid var(--line);background:var(--surface)}
header h1{margin:0;font-size:20px;letter-spacing:-.01em}header p{margin:6px 0 0;color:var(--ink2);max-width:900px}
.layout{display:grid;grid-template-columns:320px minmax(0,1fr);min-height:calc(100vh - 90px)}
aside{border-right:1px solid var(--line);background:var(--surface);display:flex;flex-direction:column;
position:sticky;top:0;height:100vh}
.tools{padding:12px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
input[type=search]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
.seg{display:flex;gap:4px;flex-wrap:wrap}
.seg button,.chip{border:1px solid var(--line);background:var(--bg);border-radius:999px;padding:3px 10px;font-size:13px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--sel);border-color:var(--accent)}
label.chk{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2);cursor:pointer}
#list{overflow:auto;flex:1;padding:4px 0;margin:0;list-style:none;position:relative}
#list li{padding:8px 12px 8px 14px;cursor:pointer;border-left:3px solid transparent;font-size:14px;line-height:1.35}
#list li:hover{background:var(--hover)}#list li[aria-selected=true]{background:var(--sel);border-left-color:var(--accent)}
#list .meta{display:block;font-size:12px;color:var(--muted);margin-top:2px}
.count{font-size:12px;color:var(--muted)}
main{padding:20px 24px 40px;min-width:0}
.q{font-size:19px;font-weight:600;margin:0 0 6px}.qmeta{color:var(--ink2);font-size:13px;margin-bottom:12px}
.sys{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
.chip{display:inline-flex;align-items:center;gap:6px}.chip[aria-pressed=false]{opacity:.5}
.dot{width:9px;height:9px;border-radius:50%;flex:none}.dot.yandex{background:var(--yandex)}.dot.qwen{background:var(--qwen)}
.dot.hollow{background:transparent;border:2px solid currentColor}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
.card h3{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--ink2);display:flex;align-items:center;gap:8px}
.badges{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.b{font-size:12px;border-radius:6px;padding:1px 7px;background:var(--hover);color:var(--ink2)}
.b.warn{background:var(--warn-bg);color:var(--warn)}.b.ok{background:var(--ok-bg);color:var(--ok)}
.rag{font-size:12px;color:var(--muted);margin:0 0 8px}
.md{overflow-wrap:anywhere;font-size:14px}.md h4{margin:6px 0 6px;font-size:16px}.md h5{margin:12px 0 4px;font-size:14px;color:var(--ink2)}
.md p{margin:4px 0}.md ul,.md ol{margin:4px 0;padding-left:22px}.md li{margin:2px 0}
.empty{color:var(--muted);padding:40px 0}
footer{color:var(--muted);font-size:12px;margin-top:28px}
@media (max-width:860px){.layout{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
#list{max-height:38vh}main{padding:16px}header{padding:16px}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Шеф-LLM: галерея ответов</h1>
<p>200 запросов из оценки и ответы восьми систем: YandexGPT-5 Lite 8B и Qwen2.5 7B — исходные и дообученные (LoRA),
без поиска и с поиском по базе рецептов (RAG). Декодирование жадное, ответы не отбирались. Значки — автометрики из отчёта.
<a href="__REPO__">Код и отчёт</a></p></header>
<div class="layout">
<aside><div class="tools">
<input type="search" id="q" placeholder="Поиск: борщ, творог, десерт…" aria-label="Поиск по запросам">
<div class="seg" id="types"></div>
<label class="chk"><input type="checkbox" id="onlybad"> только где хоть одна система ошиблась</label>
<span class="count" id="count"></span></div>
<ul id="list" role="listbox" aria-label="Запросы"></ul></aside>
<main id="main"></main></div>
<script>
const DATA = __DATA__;
const TYPES = [["all","Все"],["dish","Блюда"],["ingredients","Из продуктов"],["category","Категории"]];
const S = {type:"all", q:"", onlybad:false, id:null, hidden:new Set()};
try { const h = JSON.parse(localStorage.getItem("gallery:hidden") || "[]"); h.forEach(x => S.hidden.add(x)); } catch (e) {}
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function md(t){
  const out=[]; let list=null; const close=()=>{ if(list){ out.push("</"+list+">"); list=null; } };
  for (const raw of (t||"").split("\n")) {
    const l = raw.trim(); if (!l) { close(); continue; } let m;
    if ((m = l.match(/^#\s+(.*)/))) { close(); out.push("<h4>"+inl(m[1])+"</h4>"); }
    else if ((m = l.match(/^#{2,6}\s+(.*)/))) { close(); out.push("<h5>"+inl(m[1])+"</h5>"); }
    else if ((m = l.match(/^[-*•]\s+(.*)/))) { if(list!=="ul"){ close(); out.push("<ul>"); list="ul"; } out.push("<li>"+inl(m[1])+"</li>"); }
    else if ((m = l.match(/^\d+[.)]\s+(.*)/))) { if(list!=="ol"){ close(); out.push("<ol>"); list="ol"; } out.push("<li>"+inl(m[1])+"</li>"); }
    else { close(); out.push("<p>"+inl(l)+"</p>"); }
  }
  close(); return out.join("");
}
function inl(s){ return esc(s).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>"); }
const fam = k => (DATA.systems.find(s => s.key===k)||{}).family || "";
function bad(p){ return Object.values(p.a).some(f => (f.key && f.key[0]<f.key[1]) || (f.req && f.req[0]<f.req[1]) || f.truncated); }
function visible(){
  const q = S.q.toLowerCase();
  return DATA.prompts.filter(p => (S.type==="all" || p.type===S.type) && (!q || p.prompt.toLowerCase().includes(q))
    && (!S.onlybad || bad(p)));
}
function renderList(){
  const v = visible(), ul = document.getElementById("list");
  document.getElementById("count").textContent = v.length + " из " + DATA.prompts.length + " запросов";
  ul.innerHTML = v.map(p => '<li role="option" data-id="'+p.id+'" aria-selected="'+(p.id===S.id)+'">'+esc(p.title)+
    '<span class="meta">'+esc(p.typeLabel)+(bad(p)?" · есть ошибки":"")+'</span></li>').join("");
  if (v.length && !v.some(p => p.id===S.id)) select(v[0].id, false);
  else if (!v.length) { S.id=null; renderMain(); }
}
function badges(f){
  const b = [];
  if (f.key) b.push('<span class="b '+(f.key[0]<f.key[1]?"warn":"ok")+'">ключевые ингредиенты '+f.key[0]+'/'+f.key[1]+'</span>');
  if (f.req) b.push('<span class="b '+(f.req[0]<f.req[1]?"warn":"ok")+'">продукты из запроса '+f.req[0]+'/'+f.req[1]+'</span>');
  if (f.truncated) b.push('<span class="b warn">оборван</span>');
  if (f.chatter) b.push('<span class="b warn">«болтовня»</span>');
  if (f.nostruct) b.push('<span class="b warn">нет структуры</span>');
  b.push('<span class="b">'+f.chars+' симв.</span>');
  return b.join("");
}
function chip(s){
  const on = !S.hidden.has(s.key);
  return '<button class="chip" aria-pressed="'+on+'" data-sys="'+s.key+'"><span class="dot '+s.family+(s.ft?"":" hollow")+
    '" style="color:var(--'+s.family+')"></span>'+esc(s.label)+'</button>';
}
function renderMain(){
  const m = document.getElementById("main");
  const p = DATA.prompts.find(x => x.id===S.id);
  if (!p) { m.innerHTML = '<div class="empty">Ничего не найдено — измените фильтр.</div>'; return; }
  let meta = esc(p.typeLabel);
  if (p.must_have) meta += " · проверяемые ключевые ингредиенты: " + esc(p.must_have.join(", "));
  if (p.requested) meta += " · продукты из запроса: " + esc(p.requested.join(", "));
  const cards = DATA.systems.filter(s => !S.hidden.has(s.key) && p.a[s.key]).map(s => {
    const f = p.a[s.key];
    return '<article class="card"><h3><span class="dot '+s.family+(s.ft?"":" hollow")+'" style="color:var(--'+s.family+')"></span>'+
      esc(s.label)+'</h3><div class="badges">'+badges(f)+'</div>'+
      (f.rag ? '<p class="rag">Найдено в базе: '+f.rag.map(esc).join("; ")+'</p>' : "")+
      '<div class="md">'+md(f.text)+'</div></article>';
  }).join("");
  m.innerHTML = '<h2 class="q">'+esc(p.prompt)+'</h2><div class="qmeta">'+meta+'</div>'+
    '<div class="sys" aria-label="Показывать системы">'+DATA.systems.map(chip).join("")+'</div>'+
    (cards ? '<div class="grid">'+cards+'</div>' : '<div class="empty">Все системы скрыты — включите хотя бы одну.</div>')+
    '<footer>Ответы сгенерированы моделями и не проверялись поваром. Значки — эвристики: проверяют наличие слов, а не кулинарную правильность.</footer>';
}
function select(id, scroll=true){
  S.id = id; history.replaceState(null, "", "#"+id);
  document.querySelectorAll("#list li").forEach(li => li.setAttribute("aria-selected", li.dataset.id===id));
  renderMain();
  if (scroll) {  // scroll the list only, never the page
    const ul = document.getElementById("list"), li = ul.querySelector('li[data-id="'+id+'"]');
    if (li && (li.offsetTop < ul.scrollTop || li.offsetTop + li.offsetHeight > ul.scrollTop + ul.clientHeight))
      ul.scrollTop = li.offsetTop - ul.clientHeight / 2;
  }
}
document.getElementById("types").innerHTML = TYPES.map(([k,l]) => '<button data-t="'+k+'" aria-pressed="'+(k===S.type)+'">'+l+'</button>').join("");
document.getElementById("types").onclick = e => { const b = e.target.closest("button"); if (!b) return;
  S.type = b.dataset.t; document.querySelectorAll("#types button").forEach(x => x.setAttribute("aria-pressed", x===b)); renderList(); };
document.getElementById("q").oninput = e => { S.q = e.target.value; renderList(); };
document.getElementById("onlybad").onchange = e => { S.onlybad = e.target.checked; renderList(); };
document.getElementById("list").onclick = e => { const li = e.target.closest("li"); if (li) select(li.dataset.id, false); };
document.getElementById("main").onclick = e => { const b = e.target.closest("[data-sys]"); if (!b) return;
  const k = b.dataset.sys; S.hidden.has(k) ? S.hidden.delete(k) : S.hidden.add(k);
  try { localStorage.setItem("gallery:hidden", JSON.stringify([...S.hidden])); } catch (e) {}
  renderMain(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const v = visible(), i = v.findIndex(p => p.id===S.id);
  if ((e.key==="ArrowDown"||e.key==="j") && i < v.length-1) { e.preventDefault(); select(v[i+1].id); }
  if ((e.key==="ArrowUp"||e.key==="k") && i > 0) { e.preventDefault(); select(v[i-1].id); }
});
const start = decodeURIComponent(location.hash.slice(1));
S.id = DATA.prompts.some(p => p.id===start) ? start : DATA.prompts[0].id;
renderList(); select(S.id);
</script></body></html>
"""

TYPE_LABEL = {"dish": "Классическое блюдо", "ingredients": "Из продуктов", "category": "Категория"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "eval/results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/index.html"))
    ap.add_argument("--repo", default="https://github.com/iseequeue/recipe-llm")
    a = ap.parse_args()

    found = {os.path.basename(p)[:-6]: p for p in glob.glob(os.path.join(a.results, "*.jsonl"))}
    names = [n for n in ORDER if n in found] + sorted(set(found) - set(ORDER))
    if not names:
        raise SystemExit(f"No results in {a.results}")
    rows = {n: {r["id"]: r for r in load(found[n])} for n in names}
    ids = list(rows[names[0]])  # prompt order of the eval file
    for n in names[1:]:
        ids += [i for i in rows[n] if i not in ids]

    prompts = []
    for pid in ids:
        ref = next(rows[n][pid] for n in names if pid in rows[n])
        title = ref.get("dish") or (", ".join(ref["requested"]) if ref.get("requested") else ref["prompt"])
        p = {"id": pid, "type": ref.get("type", ""), "typeLabel": TYPE_LABEL.get(ref.get("type"), ""),
             "prompt": ref["prompt"], "title": title, "a": {}}
        if ref.get("must_have"):
            p["must_have"] = ["/".join(x + "…" for x in m.replace("\\b", "").split("|")[:3]) for m in ref["must_have"]]
        if ref.get("requested"):
            p["requested"] = ref["requested"]
        for n in names:
            if pid in rows[n]:
                r = rows[n][pid]
                p["a"][n] = {"text": r.get("response") or "", **flags(r)}
        prompts.append(p)
    systems = [{"key": n, "label": label(n), "family": "yandex" if n.startswith("yandex") else "qwen",
                "ft": "-ft" in n} for n in names]
    data = json.dumps({"systems": systems, "prompts": prompts}, ensure_ascii=False, separators=(",", ":"))
    html = PAGE.replace("__DATA__", data.replace("</", "<\\/")).replace("__REPO__", a.repo)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{len(prompts)} prompts x {len(names)} systems -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()