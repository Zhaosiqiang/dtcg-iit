# DTCG: IEEE TII manuscript and reproducibility package

This repository contains the Regular Research Paper package for IEEE Transactions on Industrial Informatics (TII), titled “Drift-Triggered Conformal Quality Gating for Industrial IoT under Production Shift.” The method couples an unlabeled batch drift trigger, class-conditional conformal calibration, a first-shift hold, delayed-label updates, and an auditable decision trace.

The paper is aerospace-oriented and is prepared to support an Airbus internship portfolio, but it does not use Airbus proprietary data and does not claim aircraft-production validation. The experiments use the public UCI SECOM semiconductor manufacturing benchmark and UCI AI4I 2020 Predictive Maintenance benchmark.

## Requirements

The experiment requires Python 3.10 or later. The tested environment used Python 3.12 with NumPy, pandas, SciPy, scikit-learn, and Matplotlib. Install the dependencies in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Run the complete chronological experiment from the repository root:

```bash
python run_experiment.py
```

The default run uses five fixed seeds and regenerates the main metrics, ablations, sensitivity sweeps, statistical tests, stable-prefix stress test, weighted-baseline diagnostics, decision trace, manifest, and all figures. In the tested Miniconda environment, the run completes within the intended short reproducibility workflow and requires no proprietary data or network access.

Key outputs include:

- `results/method_summary.csv`: aggregate method comparison used by the main results table;
- `results/seed_batch_metrics.csv`: seed--batch metrics used for paired tests;
- `results/decision_trace.csv`: auditable policy states, thresholds, actions, and shift flags;
- `results/stable_prefix_analysis.csv`: stable-prefix and post-shift AUC statistics;
- `results/sensitivity_*.csv`: alpha, drift threshold, batch-size, and class-weight sweeps;
- `figures/*.pdf` and `figures/*.png`: vector and high-resolution figure exports.

The sensitivity exports use five fixed seeds for every parameter value. `sensitivity_class_weight.csv` therefore reports a seed-level standard deviation for both datasets, with no blank standard-deviation cells after a successful run.

## Compile the manuscript

The manuscript uses the included IEEEtran class and bibliography style. With Tectonic installed, compile from the repository root:

```bash
tectonic --keep-logs --keep-intermediates main.tex
```

The resulting `main.pdf` is an 8-page Letter-format manuscript. The local quality checks verify the page count, references, figures, tables, algorithm, author metadata, and log warnings after compilation.

## Repository structure

```text
main.tex                 manuscript source
main.pdf                 compiled manuscript
references.bib           cited bibliography
run_experiment.py        end-to-end experiment and plotting script
requirements.txt         Python dependencies
data/                    frozen public-data inputs
results/                 CSV metrics, tests, traces, and manifest
figures/                 PDF and PNG figures
HIGHLIGHTS.md            TII highlights bullets
AUTHOR_METADATA.md       author, affiliation, ORCID, and CRediT record
COVER_LETTER_TEMPLATE.md cover-letter text
```

## Data and citation

The public data sources are UCI SECOM (DOI `10.24432/C54305`) and UCI AI4I 2020 Predictive Maintenance (DOI `10.24432/C52G8C`). The repository contains the frozen inputs used for the reported experiments.

Suggested citation:

> Siqiang Zhao and Fengxiang Zhang, “Drift-Triggered Conformal Quality Gating for Industrial IoT under Production Shift.”

The public reproducibility repository is [github.com/Zhaosiqiang/dtcg-iit](https://github.com/Zhaosiqiang/dtcg-iit). A Zenodo DOI has not been minted in this release. Before submission, the corresponding author must confirm authorship and ORCID information, complete the IEEE Author Portal checks, run the IEEE PDF Checker, and independently review claims, data rights, authorship, and simultaneous-submission status.
