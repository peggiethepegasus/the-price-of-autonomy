# The Price of Autonomy
**Accuracy, cost and time trade-offs of LLM agents building a predictive model**

Owen Chong | CSCI E-89 Deep Learning, Harvard Extension School | Fall 2026

Autonomous LLM agents were given the same short brief and the same raw hotel-booking data and asked to build a **booking cancellation / no-show ("at-risk") classifier** using only a terminal. An independent evaluator scored every deliverable on a hidden, later test period against a tuned XGBoost benchmark, while tokens, cost, time, rule compliance and two hidden data traps (leaky columns and duplicate rows) were recorded.

* 6 hosted models (Anthropic Claude Haiku / Sonnet / Opus, OpenAI GPT Luna / Terra / Sol) x 3 reasoning-effort levels x 3 replicates = **54 runs**, plus 6 runs of a small local model (Google Gemma 4).

## Start here
Open **`ThePriceOfAutonomy_Replication.ipynb`**.

* **Part A** replicates the analysis (data, cleaning, Optuna-XGBoost benchmark, independent evaluator, ablation, re-scoring of saved agent deliverables, campaign charts). Python only, no API keys, no cost, about 5 minutes.
* **Part B** replicates the agent experiment (brief, budgets, run plan, isolated run folders, optional launch of a real agent with [Hermes](https://hermes-agent.nousresearch.com/docs), scoring and rule audit). The step that launches an agent is **off by default** because it uses a paid LLM.

## Quick start
```bash
git clone https://github.com/<YOUR-USERNAME>/the-price-of-autonomy.git
cd the-price-of-autonomy
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab ThePriceOfAutonomy_Replication.ipynb
```
Python 3.13 was used. For a faster run of the benchmark tuning: `N_TRIALS=15 jupyter lab ...` (the benchmark score will then differ slightly from the published one).

## Repository layout
| Path | Contents |
|---|---|
| `ThePriceOfAutonomy_Replication.ipynb` | the demo: explanations, code and results |
| `data/train_val.db` | SQLite: tables `train` (83,067 rows) and `validation` (17,730 rows) |
| `data/test.db` | SQLite: `features` (17,752 rows, post-outcome columns blank) and `labels` (17,699 unique bookings) |
| `src/cleaning.py` | cleaning and the 24 model inputs |
| `src/evaluator.py` | the independent evaluator (validation, leak test, hidden test, bootstrap interval) |
| `src/harness.py` | run folders, Hermes command builder, time limit, rule auditor |
| `agent_harness/` | the frozen brief, budgets, run plan (60 runs) and brief hash |
| `example_runs/` | three real agent deliverables (B38, B13, B09) with their published scores |
| `results/` | campaign tables (per-run, per-setting averages, hypothesis scorecard, cost recomputation, B38 ablation) and the published benchmark metrics |

## Data
Hotel-booking records of a City hotel and an Airport hotel of a Singapore group, **provided by AI Singapore** (118,549 bookings, arrivals August 2023 to August 2025). The data is split by time: train = oldest 70%, validation = next 15%, test = latest 15%. Booking IDs are random, and two ambiguous columns were removed. All columns are stored as text with deliberately messy formatting. The files are small (about 20 MB in total), so the **full data is included** in this repository.

## Reproducibility notes
* Random seed 42 everywhere a seed is accepted; Optuna TPE sampler seed 42, 60 trials, 4-fold time-ordered cross-validation.
* Agent runs are **not deterministic**; the study reports spread over three replicates and makes no claim of statistical significance.
* Token counts differ between vendors because of caching, so they are compared within a vendor family. Costs are list-price equivalents.
* Results were produced with the library versions in `requirements.txt`.

## Licence and attribution
Code: MIT licence (see `LICENSE`). Data: AI Singapore hotel-booking records, redistributed with the data owner's permission; please cite AI Singapore if you reuse it.
