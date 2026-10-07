# Dashboard & reports

Two ways to view outputs. Both read the `results/` folder; neither changes any
experiment logic.

## 1. Static HTML report (no dependencies, shareable)

Bakes every table and figure under a results folder into one self-contained
`report.html` (open it in any browser; good for a supervisor or a thesis
appendix):

```bash
python -m scripts.make_report                                   # results/ -> results/report.html
python -m scripts.make_report --results results/campaign/laptop
python -m scripts.make_report --results results/experiments --out report.html
```

## 2. Interactive dashboard (Streamlit, live run console)

A local web interface: browse results (tables + charts + figures) and launch
runs from a form, streaming the console output live.

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Then in the browser: the **Results** tab shows tables and bar charts, **Figures**
shows the PNGs, and **Run** lets you launch `src.run` on a config or an
experiment (E1–E6) and watch the `[prequential]` heartbeat stream into a console
pane. Run it from the `03_Prototype` folder so paths resolve.

> The dashboard is a viva / demo and exploration aid. It shells out to the same
> commands you would type by hand, so results are identical to a terminal run.
