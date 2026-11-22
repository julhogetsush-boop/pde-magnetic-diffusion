# Verified magnetic diffusion in a conducting slab

Team 13 · Numerical PDEs Project 3

The two finite-difference methods are FTCS and Crank–Nicolson. The experiments compare the numerical field with exact modes, the continuous startup series and the settled finite-slab transfer function. All quantities in the numerical tables are dimensionless except phase in degrees and timing in seconds.

Python 3.11 or newer is required. From the extracted folder, install the three numerical packages, check the solvers, and reproduce all five report figures and the CSV data.

```text
python -m pip install -r requirements.txt
python code/run_all.py --check
python code/run_all.py
```

The results are written to `results/`. `summary.json` records the two crossings, the numerical agreement checks and the cost comparison. The CSVs give grid spacing, time steps and reference errors. In `crossings.csv`, `time_setting` means FTCS's ratio `k/h²` or CN's steps per period. `SD` is exact in time on the finite-difference grid. Wall-clock times vary across machines and runs.

`code/model.py` contains the grid operator, the boundary forcing, the updates and the references. `code/run_all.py` produces the experiments and plots. The editable report source is `report.tex`. After running the experiments, compile it from this folder with `latexmk -pdf report.tex` or `tectonic report.tex`.

`course_templates/` preserves the original course PDFs and LaTeX files for the report, individual contribution statement, problem definition, code review, submission and visualisation. The individual contribution pages in the report reserve space for each member's own evidence and signature.
