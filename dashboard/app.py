"""Interactive experiments dashboard (Streamlit) for CS7P01NU.

A local web application over the prototype. It lets a user (or examiner) run
every experiment the project needs from a curated catalogue, watch the console
output and status live, browse results as tables and charts, view the generated
figures, and export any result set to CSV, Excel or a PDF report.

Run from the 03_Prototype folder:
    pip install -r requirements-dashboard.txt
    streamlit run dashboard/app.py

It changes no experiment logic; it shells out to the same commands one would run
by hand, and reads the results/ folder.
"""
from __future__ import annotations

import glob
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]  # 03_Prototype
DISPLAY_ROOT = "/".join(BASE.parts[-2:])  # show only project-relative root, not the machine/user path
_HOME = str(Path.home())
def _sanitize(txt):
    """Strip absolute machine paths from anything shown in the UI."""
    return txt.replace(str(BASE), ".").replace(_HOME, "~")
st.set_page_config(page_title="Learned Context Construction — Experiments Dashboard", layout="wide")

# ------------------------------------------------------------------ header
st.title("Learned Context Construction — Experiments Dashboard")
st.caption(
    "In-context tabular models under concept drift and class imbalance (CS7P01NU). "
    "Run experiments, watch progress live, explore results, and export to CSV / Excel / PDF."
)
st.caption(f"Project root: {DISPLAY_ROOT}")


# ------------------------------------------------------------------ helpers
def find_result_dirs():
    dirs = set()
    for p in BASE.glob("results/**/*"):
        if p.suffix.lower() in {".csv", ".json", ".png"}:
            dirs.add(str(p.parent.relative_to(BASE)))
    return sorted(dirs) or ["results"]


def approach_col(df):
    for c in ("approach", "selector", "reward", "length", "config"):
        if c in df.columns:
            return c
    return df.columns[0]


def metric_col(df):
    for c in ("balanced_accuracy_mean", "balanced_accuracy", "bal_acc_mean", "bal_acc",
              "overall", "policy_bal_acc"):
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def to_excel_bytes(frames):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in frames.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
    return buf.getvalue()


def to_pdf_bytes(csv_paths, png_paths):
    """Assemble a PDF report: cover, one page per table, one page per figure."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(0.5, 0.62, "Learned Context Construction", ha="center", size=22, weight="bold")
        cover.text(0.5, 0.55, "Experiment results report", ha="center", size=14)
        cover.text(0.5, 0.48, "CS7P01NU MSc Project", ha="center", size=11, color="#555")
        pdf.savefig(cover); plt.close(cover)
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df = df.iloc[:24, :9]
            if df.shape[0] == 0 or df.shape[1] == 0:  # skip empty/header-only tables
                continue
            try:
                df = df.round(4)
            except Exception:
                pass
            try:
                fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111); ax.axis("off")
                ax.set_title(os.path.basename(path), loc="left", weight="bold", pad=14)
                tbl = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                               loc="center", cellLoc="center")
                tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.3)
                pdf.savefig(fig, bbox_inches="tight")
            except Exception:
                pass
            finally:
                plt.close("all")
        for path in png_paths:
            try:
                img = plt.imread(path)
            except Exception:
                continue
            fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111); ax.axis("off")
            ax.set_title(os.path.basename(path), loc="left", weight="bold", pad=10)
            ax.imshow(img)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


# curated catalogue of every experiment the project needs
CATALOG = [
    ("Setup — TabPFN probe (~3 min)", "Confirms TabPFN is installed and produces predictions distinct from the reference substrate.",
     ["-m", "scripts.experiments.probe_tabpfn"], False),
    ("Setup — component tests (seconds)", "Runs the automated suite: signals, baselines, drift injector, and the policy degeneracy proof.",
     ["-m", "tests.test_components"], False),
    ("Month-2 checkpoint — Electricity (TabPFN + sliding)", "The go/no-go reproduction on real data; expects balanced accuracy ~0.96.",
     ["-m", "src.run", "--config", "config/electricity_sliding.yaml"], True),
    ("E1 — convergence vs context length", "How the policy-baseline gap narrows as context grows.",
     ["-m", "scripts.experiments.exp_e1_convergence"], True),
    ("E2 — optimiser / hyperparameter sweep", "Finds the strongest policy-gradient configuration.",
     ["-m", "scripts.experiments.exp_e2_optimizer", "--imbalance", "0.15"], True),
    ("E3 — reward design", "Compares instance-level vs windowed, class-balanced rewards.",
     ["-m", "scripts.experiments.exp_e3_reward", "--imbalance", "0.15"], True),
    ("E4 — ceiling / headroom analysis", "Whether a fixed blend can beat the best single axis (headroom).",
     ["-m", "scripts.experiments.exp_e4_ceiling", "--imbalance", "0.15"], True),
    ("E5 — shifting-optimal-axis regime", "The regime where an adaptive policy should win; per-segment analysis.",
     ["-m", "scripts.experiments.exp_e5_shifting"], True),
    ("E6 — stability & weight-collapse", "Weight-trajectory volatility and latency distribution.",
     ["-m", "scripts.experiments.exp_e6_stability"], True),
    ("Campaign — laptop (quick wiring check)", "Fast smoke sweep of the full campaign (reference classifier).",
     ["-m", "scripts.run_campaign", "--config", "config/campaign_laptop.yaml", "--quick"], False),
    ("Report — build HTML results report", "Bakes every table and figure under results/ into one report.html.",
     ["-m", "scripts.make_report"], False),
]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Results folder")
result_dirs = find_result_dirs()
sel_dir = st.sidebar.selectbox("Folder", result_dirs,
                               index=result_dirs.index("results") if "results" in result_dirs else 0)
if st.sidebar.button("↻ Refresh"):
    st.rerun()
full_dir = BASE / sel_dir
st.sidebar.markdown("---")
st.sidebar.caption("Exports and figures reflect the selected folder.")

tab_results, tab_figures, tab_run, tab_export = st.tabs(
    ["📊 Results", "🖼 Figures", "▶ Run experiments", "⬇ Export"])

# ------------------------------------------------------------------ results
with tab_results:
    csvs = sorted(glob.glob(str(full_dir / "*.csv")))
    if not csvs:
        st.info("No CSV files in this folder yet. Run an experiment (▶ tab) or pick another folder.")
    for path in csvs:
        st.subheader(os.path.basename(path))
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            st.warning(f"could not read: {e}")
            continue
        st.dataframe(df, width="stretch", hide_index=True)
        ac, mc = approach_col(df), metric_col(df)
        if mc and ac in df.columns and 1 < df[ac].nunique() <= 30 and "t" not in df.columns:
            try:
                st.bar_chart(df[[ac, mc]].dropna().set_index(ac).sort_values(mc, ascending=False), height=280)
            except Exception:
                pass
        if "t" in df.columns:
            ycols = [c for c in ("windowed_accuracy", "balanced_accuracy", "windowed_auprc", "ctx_pos_rate")
                     if c in df.columns]
            if ycols:
                try:
                    st.line_chart(df.set_index("t")[ycols], height=280)
                    st.caption("Prequential metrics / context composition over the stream.")
                except Exception:
                    pass
        st.download_button("⬇ CSV", data=df.to_csv(index=False).encode(),
                           file_name=os.path.basename(path), mime="text/csv", key=f"csv-{path}")

# ------------------------------------------------------------------ figures
with tab_figures:
    pngs = sorted(glob.glob(str(full_dir / "*.png")))
    if not pngs:
        st.info("No figures in this folder yet.")
    for path in pngs:
        st.image(path, caption=os.path.basename(path), width="stretch")
        with open(path, "rb") as fh:
            st.download_button("⬇ PNG", data=fh.read(), file_name=os.path.basename(path),
                               mime="image/png", key=f"png-{path}")

# ------------------------------------------------------------------ run
with tab_run:
    st.markdown("Run any experiment the project needs and watch its output live.")
    labels = [c[0] for c in CATALOG]
    idx = st.selectbox("Experiment", range(len(labels)), format_func=lambda i: labels[i])
    label, desc, base_cmd, substrate_aware = CATALOG[idx]
    st.info(desc)
    cmd = [sys.executable] + list(base_cmd)
    cols = st.columns(3)
    if substrate_aware:
        sub = cols[0].radio("Substrate", ["reference (fast)", "TabPFN (real, slow)"], key="sub")
        if sub.startswith("TabPFN"):
            cmd += ["--tabpfn"]
        steps = cols[1].number_input("--max-steps (0 = default)", min_value=0, value=0, step=1000)
        if steps and "--config" not in base_cmd:
            cmd += ["--max-steps", str(int(steps))]
    st.code(" ".join(os.path.relpath(c, BASE) if str(c).startswith(str(BASE)) else str(c) for c in cmd), language="bash")

    if st.button("▶ Run", type="primary"):
        status = st.status(f"Running: {label}", expanded=True)
        console = status.empty()
        lines = []
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                lines.append(_sanitize(line.rstrip("\n")))
                console.code("\n".join(lines[-400:]), language="text")
            proc.wait()
            dt = time.time() - t0
            if proc.returncode == 0:
                status.update(label=f"✅ {label} — done in {dt:.0f}s", state="complete")
            else:
                status.update(label=f"❌ {label} — exit {proc.returncode} ({dt:.0f}s)", state="error")
        except Exception as e:  # noqa: BLE001
            status.update(label=f"❌ failed to launch: {e}", state="error")
        st.caption("Switch to the Results / Figures / Export tabs and Refresh to see new outputs.")

# ------------------------------------------------------------------ export
with tab_export:
    st.markdown(f"Export everything in **{sel_dir}**.")
    csvs = sorted(glob.glob(str(full_dir / "*.csv")))
    pngs = sorted(glob.glob(str(full_dir / "*.png")))
    st.write(f"{len(csvs)} table(s) and {len(pngs)} figure(s) in this folder.")
    if csvs:
        frames = {}
        for p in csvs:
            try:
                frames[os.path.splitext(os.path.basename(p))[0]] = pd.read_csv(p)
            except Exception:
                pass
        if frames:
            try:
                import openpyxl  # noqa: F401
                st.download_button("⬇ Excel workbook (all tables, one sheet each)",
                                   data=to_excel_bytes(frames),
                                   file_name=f"{Path(sel_dir).name}_results.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except ImportError:
                st.info("Excel export needs the openpyxl package. Install it with:  pip install openpyxl  "
                        "(CSV and PDF export work without it).")
    if csvs or pngs:
        try:
            st.download_button("⬇ PDF report (tables + figures)",
                               data=to_pdf_bytes(csvs, pngs),
                               file_name=f"{Path(sel_dir).name}_report.pdf",
                               mime="application/pdf")
        except Exception as e:  # noqa: BLE001
            st.warning(f"PDF report could not be built for this folder: {e}")
    if not (csvs or pngs):
        st.info("Nothing to export in this folder yet.")
