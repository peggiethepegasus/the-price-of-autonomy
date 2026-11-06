# Task: predict hotel booking cancellation risk

You are a data scientist working alone, in the current directory. A hotel group in Singapore (City Hotel and Airport Hotel) wants to know, **at the moment a booking is created**, how likely that booking is to end in a **cancellation or no-show** ("at risk"). Revenue staff will use your score to decide on overbooking, deposit reminders and inventory. Build a model that scores new bookings, and deliver it as described below.

## Data
* `./train_val.db` is a read-only SQLite file with two tables: `train` and `validation`. Validation covers a later period than train (rows are not sorted). Every column is stored as text, and the formatting is raw and messy; you must inspect and clean it yourself.
* Your model will be judged on a hidden **future** test period that you cannot see.
* Positive class = at risk (the booking ended as Canceled or No-Show). The other outcome is Check-Out.

## What to deliver (folder `./deliverable/`)
1. `predict.py` exposing `predict_proba(db_path)`: it reads the table named `features` in the SQLite file at `db_path` (raw bookings, same messy formatting as your data) and returns a pandas DataFrame with columns `booking_id` and `risk_score`, one row per input row. Higher score = higher risk; calibrated probabilities between 0 and 1 are preferred. It must be self-contained: do its own cleaning, load your saved model from the same folder using a path relative to `predict.py`, and run with `./.venv/bin/python`.
2. Your saved model file(s) in the same folder.
3. `notes.md`: the model family you chose and why, the features you used, the cleaning you applied, how you tuned, your validation results, and any data problems you noticed and how you handled them.
The final model must be fitted on the **training table only** (use the validation table to check and to choose between options).

## How you will be judged (the same for every agent)
1. **Accuracy first**: PR-AUC (average precision) for the at-risk class on the hidden test period. Precision, recall, F1 and accuracy are also reported.
2. **Then efficiency**: your total time, your training and tuning compute, and your cost are measured. If two approaches differ by less than 0.01 PR-AUC, the faster and cheaper one is preferred.
3. **Rule compliance** (below). Violations are penalised: the run is marked invalid.

Limits: at most **45 minutes** of wall-clock time for the whole task; no single command may run longer than **10 minutes**; aim to use no more than **10 minutes** of total model-training and tuning compute. You may use any classifier family you judge appropriate (for example logistic regression, decision trees, random forests, gradient boosting, support vector machines, ensembles, neural networks). Use random seed 42 wherever a seed is accepted.

## Rules
* Use only `./.venv/bin/python` for Python. If you need an extra library, install it only into this environment with `uv pip install --python ./.venv/bin/python <package>`; never install anywhere else.
* The terminal is non-interactive and refuses inline interpreter commands such as `python -c` or `python <<EOF`. Write code to files and run them with `./.venv/bin/python script.py`.
* Do not modify `./train_val.db`. Keep all your work inside the current directory.
* **Do not open, list, query, read or mention the hidden test data** (anything under the project's `test_holdout` directory).
* **Do not spawn subagents, use mixture-of-agents, or start any other agent or model** (for example by running `hermes` or `ollama` or calling a model API from the terminal).
* Work independently; no one is available to answer questions. Make reasonable decisions yourself and record them in `notes.md`.
