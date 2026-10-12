# The Effect of Generative AI-Assisted Programming on Critical Thinking and Academic Performance of Undergraduate IT Students

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Status](https://img.shields.io/badge/status-pipeline%20skeleton-orange.svg)

**Authors:** Adi Irgimbayev, Alan Shmyrin
**Repository:** <https://github.com/adiirgimbayevv/Methods_Adi_Alan>

## Overview

Generative AI tools such as ChatGPT are now common in programming courses, but it is unclear whether they strengthen
or weaken students' reasoning. This project examines how the **intensity** and the **manner** of generative AI use are
related to **critical thinking** and **academic performance** among undergraduate IT students.

This repository contains the reproducible analysis pipeline: a configurable, containerized script that reads a
student-level table, checks data-quality guardrails, tests the study hypotheses with covariate adjustment and writes a
log and a metrics file. It currently runs on a **synthetic 10-row sample**; real data collection follows later.

### Research questions

| ID  | Question |
|-----|----------|
| RQ1 | How does the use of generative AI during programming tasks affect the critical thinking performance of undergraduate IT students? |
| RQ2 | What is the relationship between the intensity of generative AI use and the academic performance of undergraduate IT students? |
| RQ3 | Does the way students use generative AI (explanations/debugging vs. full-solution generation) affect their critical thinking and academic performance? |

### Primary hypothesis (RQ3)

- **H0:** After controlling for covariates, there is no difference in critical thinking score between students who mainly use GenAI for explanations/debugging feedback and students who mainly use it for full-solution generation (β<sub>mode</sub> = 0).
- **H1:** After controlling for covariates, students who mainly use GenAI for explanations/debugging feedback have a higher critical thinking score (β<sub>mode</sub> > 0).

The critical thinking outcome is measured with the Sosu Critical Thinking Disposition Scale (self-report). Results show associations, not causal effects. The full methodology (variables, metrics, baseline, limitations) is in [`docs/Task2_Methodology_Card.pdf`](docs/Task2_Methodology_Card.pdf).

## Repository structure

```
Methods_Adi_Alan/
├── README.md
├── LICENSE                     MIT
├── Dockerfile                  pinned base image + dependencies
├── requirements.txt            exact dependency versions
├── .dockerignore
├── .gitignore
├── configs/
│   └── config.yaml             all experiment parameters (paths, variables, thresholds, seed)
├── data/
│   ├── sample/sample.csv       synthetic 10-row sample (safe to commit)
│   └── raw/                    private real data, git-ignored
├── src/
│   ├── main.py                 entry point: config -> checks -> tests -> log + metrics
│   └── analysis.py             guardrails, OLS models, effect sizes, Holm correction
├── notebooks/                  exploratory analysis (optional)
├── results/                    run.log and metrics.json are written here (git-ignored)
└── docs/
    └── Task2_Methodology_Card.pdf
```

## System requirements

| Item | Requirement |
|------|-------------|
| OS | Linux, macOS or Windows (with Docker Desktop) |
| Docker | Any recent version (recommended path), **or** Python 3.12 for a local run |
| Hardware | No GPU needed; 1 CPU core and < 1 GB RAM are enough |
| Disk | About 500 MB for the Docker image |
| Network | Needed once to download the base image and packages |

## Quickstart

Run the whole pipeline on the sample data with two commands:

```bash
docker build -t genai-study .
docker run --rm -v "$(pwd)/results:/app/results" genai-study
```

The console shows the log, and `results/run.log` and `results/metrics.json` are created.

**Without Docker** (Python 3.12):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

Optional flags: `--config <path>` selects another config file; `--strict` returns exit code 1 when a guardrail fails.

> **Note on the sample.** `data/sample/sample.csv` has only 10 invented rows to prove that the pipeline runs end to end.
> Guardrails such as sample size correctly report `FAIL`, the script logs `SMOKE-TEST MODE`, and no hypothesis conclusion is
> printed. Numbers produced from the sample have no scientific meaning.

### Using your own data

1. Place a CSV with the columns listed in `configs/config.yaml` (`variables` section) under `data/raw/` (git-ignored).
2. Set `paths.data_file` in `configs/config.yaml` to that file.
3. Run the pipeline as above. **Never commit real student data.**

## Planned evaluation metrics and baseline

| Role | Metric | Details |
|------|--------|---------|
| **Primary** | Adjusted β<sub>mode</sub> on critical thinking disposition score (Sosu CTDS, self-report) | OLS with controls; 95% CI, one-sided p (α = 0.05), Cohen's d |
| Secondary | β for assignment score (RQ3b), β for AI use (RQ1), Spearman ρ and β for intensity vs. final grade (RQ2) | Holm correction for multiple tests |
| Guardrail | Cronbach's α ≥ 0.70 for the CTDS | Needs item-level columns (`variables.ct_items`) |
| Guardrail | Missing data < 10% per variable; VIF < 5 | Checked automatically |
| Guardrail | n ≥ about 100 | A priori power analysis: f² = 0.15, α = 0.05, power = 0.80 |
| Guardrail | Anonymized data | No name, e-mail or phone columns allowed |
| **Baseline** | (a) Reference groups: non-users (RQ1) and solution-generation users (RQ3); (b) null model with controls only | Gain reported as ΔR² with F-test |

## Expected output

After a run, `results/metrics.json` contains the sections `guardrails`, `primary_rq3`, `secondary` (RQ1, RQ2, RQ3b with
Holm-adjusted p-values) and `descriptives`. The log has this form (values are placeholders on real data):

```
Loaded <n> rows x 15 columns
--- Guardrails ---
sample_size_ok   PASS | FAIL
...
--- Primary hypothesis (RQ3): explanation vs solution -> critical thinking ---
beta=<value> [<ci_low>, <ci_high>], p=<value> (greater), dR2=<value>, n=<value>
```

## Reproducibility

- Exact dependency pins in `requirements.txt`; the Docker base image is pinned to `python:3.12.3-slim`.
- All parameters, including the random seed, are in `configs/config.yaml`.
- On very small samples the script automatically drops trailing controls to keep at least one residual degree of freedom and logs a warning.

## Literature base

- Zotero collection: <https://www.zotero.org/groups/6692752/methods_adi_alan/library>
- ResearchRabbit LitMap: <https://app.researchrabbit.ai/folder-shares/9e966e1f-0b0c-438c-87c5-dd65195366e1>

## Citation

```bibtex
@misc{irgimbayev_shmyrin_2026_genai,
  author       = {Irgimbayev, Adi and Shmyrin, Alan},
  title        = {The Effect of Generative AI-Assisted Programming on Critical Thinking and Academic Performance of Undergraduate IT Students},
  year         = {2026},
  howpublished = {\url{https://github.com/adiirgimbayevv/Methods_Adi_Alan}}
}
```

## License

Released under the [MIT License](LICENSE).
