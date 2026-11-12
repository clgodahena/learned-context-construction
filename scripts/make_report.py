"""Static HTML results report — a self-contained, shareable page.

Walks a results directory and bakes every CSV, JSON summary and PNG figure it
finds into a single self-contained ``report.html`` (images inlined as data URIs,
CSS inlined). No server and no dependencies beyond the standard library — just
open the file. Good for sharing and for a thesis appendix.

Usage:
  python -m scripts.make_report                                   # results/ -> results/report.html
  python -m scripts.make_report --results results/campaign/laptop
  python -m scripts.make_report --results results/experiments --out report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime
import html
import json
import os


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def csv_table_html(rows):
    if not rows:
        return "<p><em>empty</em></p>"
    head, *body = rows
    th = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    trs = []
    for r in body:
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def img_html(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(os.path.basename(path))}"/>'


def json_html(path):
    with open(path) as f:
        try:
            obj = json.load(f)
        except Exception:
            return "<pre>could not parse JSON</pre>"
    return f"<pre>{html.escape(json.dumps(obj, indent=2))}</pre>"


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1c2530;background:#fafbfc}
header{background:#33507A;color:#fff;padding:20px 32px}
header h1{margin:0;font-size:20px} header p{margin:4px 0 0;opacity:.85;font-size:13px}
nav{position:sticky;top:0;background:#eef2f7;padding:8px 32px;border-bottom:1px solid #d7dee7;font-size:13px}
nav a{margin-right:14px;color:#33507A;text-decoration:none} nav a:hover{text-decoration:underline}
main{padding:12px 32px 60px;max-width:1100px}
section{margin:26px 0;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px}
h2{font-size:16px;color:#33507A;margin:.2em 0 .6em;border-bottom:1px solid #eef2f7;padding-bottom:6px}
h3{font-size:13px;color:#4a5b70;margin:1.2em 0 .4em}
table{border-collapse:collapse;font-size:12.5px;margin:.3em 0;overflow-x:auto;display:block}
th,td{border:1px solid #dbe2ea;padding:5px 9px;text-align:left;white-space:nowrap}
th{background:#eef2f7} tr:nth-child(even) td{background:#f7f9fb}
img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin:.4em 0}
pre{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:6px;padding:10px;font-size:12px;overflow-x:auto}
.small{color:#7a8798;font-size:12px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = args.results
    out = args.out or os.path.join(root, "report.html")
    if not os.path.isdir(root):
        raise SystemExit(f"no such results directory: {root}")

    # collect files grouped by their containing folder
    groups = {}
    for dirpath, _, files in os.walk(root):
        keep = [f for f in sorted(files) if f.lower().endswith((".csv", ".json", ".png"))]
        if keep:
            groups[dirpath] = keep

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    nav, body = [], []
    for i, (d, files) in enumerate(sorted(groups.items())):
        anchor = f"g{i}"
        rel = os.path.relpath(d, root)
        label = rel if rel != "." else os.path.basename(os.path.abspath(root))
        nav.append(f'<a href="#{anchor}">{html.escape(label)}</a>')
        parts = [f'<section id="{anchor}"><h2>{html.escape(label)}</h2>']
        # figures first, then tables, then json
        for f in [x for x in files if x.endswith(".png")]:
            parts.append(f"<h3>{html.escape(f)}</h3>{img_html(os.path.join(d, f))}")
        for f in [x for x in files if x.endswith(".csv")]:
            parts.append(f"<h3>{html.escape(f)}</h3>{csv_table_html(read_csv(os.path.join(d, f)))}")
        for f in [x for x in files if x.endswith(".json")]:
            parts.append(f"<h3>{html.escape(f)}</h3>{json_html(os.path.join(d, f))}")
        parts.append("</section>")
        body.append("".join(parts))

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Results Report — CS7P01NU</title><style>{CSS}</style></head><body>
<header><h1>CS7P01NU — Results Report</h1>
<p>Learned Context Construction for In-Context Tabular Models · generated {stamp} · source: {html.escape(root)}</p></header>
<nav>{''.join(nav) if nav else '<span class="small">no result files found</span>'}</nav>
<main>{''.join(body) if body else '<p class="small">No CSV/JSON/PNG files under this folder yet — run an experiment first.</p>'}</main>
</body></html>"""

    with open(out, "w") as f:
        f.write(doc)
    n_png = sum(1 for g in groups.values() for x in g if x.endswith(".png"))
    n_csv = sum(1 for g in groups.values() for x in g if x.endswith(".csv"))
    print(f"[report] wrote {out}  ({len(groups)} folders, {n_csv} tables, {n_png} figures)")


if __name__ == "__main__":
    main()
