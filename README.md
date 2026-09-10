# IEEE TII manuscript package

This folder contains the Regular Research Paper package for IEEE Transactions on Industrial Informatics (TII), titled “Drift-Triggered Conformal Quality Gating for Industrial IoT under Production Shift.” The paper is method-first and aerospace-oriented, but it does not claim to use Airbus data.

## Deliverables

- `main.tex` and `main.pdf`: the IEEEtran manuscript, with the two-author block (Siqiang Zhao first/corresponding; Fengxiang Zhang second), the confirmed local ORCID/email metadata, declarations, Algorithm 1, two-dataset results, ablations, sensitivity analysis, statistical tests, and Industrial Implications.
- `references.bib`: 37 references; a local citation audit confirms every bibliography key is cited in `main.tex`.
- `run_experiment.py`: reproducible chronological experiment over UCI SECOM and UCI AI4I 2020 Predictive Maintenance.
- `data/`: frozen public-data inputs used by the experiment.
- `results/`: main summaries, per-seed/per-batch metrics, ablations, first-five/first-batch hold checks, sensitivities, Wilcoxon tests, stable-prefix analysis, weighted-baseline diagnostics, decision trace, and manifest.
- `figures/`: vector PDF figures plus 600-dpi PNG exports.
- `COVER_LETTER_TEMPLATE.md`: completed cover letter with no placeholder fields.
- `投稿类型决策.md`: G-01 Regular-vs-Short decision record.
- `DTCG_TII_审查_20260910/02_逐条验收记录.md`: item-by-item acceptance record for every work order in the supplied checklist.
- `AUTHOR_METADATA.md`: two-author order, affiliation assignments, email/ORCID provenance, CRediT roles, and the remaining portal-only actions.

## Reproduce the experiment and compile

From this directory:

```text
python run_experiment.py
tectonic --keep-logs --keep-intermediates main.tex
```

The manuscript uses XeLaTeX/Tectonic with `fontspec` and the installed Times New Roman family. The local `IEEEtran.cls` has a narrow XeLaTeX initialization compatibility branch; the body font remains Times New Roman. After compilation, confirm `pdfinfo main.pdf` reports 8–10 pages and confirm that `main.log` contains no `LaTeX Font Warning`.

The manuscript is not an acceptance guarantee and has not been submitted by this workspace. Before upload, the corresponding author must confirm the final author list, bind ORCID in IEEE Author Portal, provide a permanent public code/DOI link if required, run the portal’s IEEE PDF Checker, and independently review the claims, data rights, authorship, and simultaneous-submission status.

The sensitivity exports use five fixed seeds for every parameter value. In particular, `sensitivity_class_weight.csv` reports a seed-level standard deviation for both datasets; blank standard-deviation cells are not expected after running `python run_experiment.py`.
