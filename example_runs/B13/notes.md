# Booking cancellation / no-show risk: notes

## Deliverable

- `predict.py`: `predict_proba(db_path)` reads table `features`, cleans it, scores every input row and returns `booking_id`, `risk_score` (one row per input row, probabilities in [0, 1]). Model files are loaded from paths relative to the file.
- `model.txt`: LightGBM booster (text format). `model_meta.json`: feature list, categorical vocabularies, ADR cap, parameters.
- `train.py`: refits the model from the training table only and rewrites `model.txt` / `model_meta.json`. Uses the same `clean` / `encode` code as `predict.py`.

## Model family and why

Gradient-boosted decision trees (LightGBM, binary logloss). The data is tabular, mixed-type, and has strong non-linear effects (lead time, market segment, deposit type, and their interactions). Logistic regression with one-hot categoricals reached PR-AUC 0.8135 on validation. LightGBM reached about 0.85 across a broad range of settings. Training is about 3 seconds on 81.6k rows, so there was no reason to go to a heavier ensemble.

## Target

`is_canceled` is stored in six spellings (0/1, false/true, No/Yes). After normalisation it agrees exactly with `reservation_status`: Canceled and No-Show are 1 (at risk), Check-Out is 0. Train positive rate 35.1%, validation 39.3%.

## Features (all known when the booking is created)

Numeric: lead_time (clipped at 0), arrival year, month, day, day-of-week, ISO week, day-of-year (all recomputed from the date), arrival_days (days since 2023-01-01), weekend, week and total nights, adults, children, babies, total guests, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, adr (parsed, capped at 1000).

Categorical (integer-coded, vocabulary built from training with a minimum count of 50, anything else goes to an OTHER code): hotel, meal, country, market_segment, distribution_channel, deposit_type, customer_type, reserved_room_type, agent, promo_code.

Excluded on purpose:
- reservation_status, reservation_status_date: these are the outcome.
- booking_changes, total_of_special_requests: they accumulate until check-in or later, so they are not known at creation. Including them in an information-only run gave validation PR-AUC 0.881 (+0.03). I did not use them because the brief asks for a score at the moment of booking.
- assigned_room_type: assigned at check-in.
- company: only its presence/absence was checked; it had no signal (35.2% vs 35.1%) and the names are sparse.
- booking_id: used only to label output rows.

## Cleaning applied

- Target: six encodings mapped to 0/1 (see above).
- Duplicates: 244 exact duplicate rows in train and 56 in validation (identical content, repeated booking_id). Dropped in training. `predict_proba` scores every input row, duplicates included.
- Dates: month as full name or 3-letter abbreviation (Apr / April) mapped to 1..12. The raw week number is noisy (blank, '-', 'unknown'), so it is recomputed from year, month and day.
- Lead time: '14 days' style strings parsed. 206 negative values in train (48 in validation) clipped to 0.
- ADR: '138,00' comma decimals, 'SGD 172.50' prefixes, 'N/A' and blanks all parsed (blank becomes NaN; none occurred in the data). Zero ADR (complimentary stays, 1.7%) kept.
- Text categoricals: trimmed, lower-cased, whitespace collapsed, and placeholders ('', '-', NULL, NA, N/A, unknown, Undefined) mapped to 'unknown'. Spelling variants fixed: segment 62 raw spellings to 8 levels (incl. 'Onlin TA', 'Offline TA/T0', 'Offline-TA/TO'), deposit 23 to 3 ('No Depost', 'NO DEPOSIT', 'Non-Refund'), customer type ('Transiant'), meal 38 to 5, distribution channel. Country names mapped to the ISO3 codes used elsewhere ('Singapore' to 'SGP', etc.): 19 levels.
- Agent: stored as floats ('9.0'); blank means no agent and gets its own level.
- Promo code: blank means no promo and gets its own level.

## Data problems found and how they were handled

1. Corrupt ADR burst. 1,225 City Hotel rows from March 2024 have ADR 1,900 to 7,100. The hotel's 95th percentile is about 450, and validation has essentially none above 1,000 (0.01%). These rows also show a 5.9% cancel rate against 35.6% for the rest. Handling: dropped from training, and ADR capped at 1000 at scoring time. On validation, dropping them moved PR-AUC from 0.8506 to 0.8510, so the choice does not hurt.
2. Post-booking columns in the same table (see Excluded above).
3. Adults = 0 with no children or babies: 130 train rows, 33 validation rows. Kept (small, likely entry error). Party-size outliers (up to 55 adults) left as-is, since trees split on ranks and are robust to them.
4. Drift between periods. Train covers Aug 2023 to Feb 2025 and validation covers Feb to May 2025 (arrival dates). The validation positive rate is higher (39.3% vs 35.1%), with more long-lead bookings and more Groups bookings (Groups cancel at about 57-60%). The model's mean predicted risk on validation was 0.392 against an actual 0.393, so calibration held, but the hidden period could drift further.
5. Hidden-test feature layout is unknown. `predict_proba` reads only the columns it needs and ignores any outcome columns that may be present.

## Tuning

Validation was used for all choices (the brief allows this). The final model was fit on training only; it was not refit on validation.

- Grid: LightGBM, learning rate 0.03, num_leaves in {15, 31, 63}, min_data_in_leaf in {100, 300}, feature_fraction 0.8, bagging 0.8, lambda_l2 1, seed 42. Checkpoints from 100 to 1500 rounds. PR-AUC spanned 0.846 to 0.8515 across the whole grid, a flat plateau.
- Chosen: num_leaves 15, min_data_in_leaf 100, 600 rounds. PR-AUC 0.8506, within 0.001 of the best setting (num_leaves 63 or 31 with min_data 300), but about 3x cheaper. The brief prefers the cheaper model when the gap is under 0.01.
- Ablation: dropping arrival_days costs 0.003 PR-AUC (0.8480 vs 0.8506) and I kept it. The model will see dates after validation as the top bin, since trees clamp at the training range. A later test period will therefore not be learned as a trend.

## Validation results (final model, scored through `predict_proba`)

Trained on training only. Evaluated on validation after dedupe, n=17,674:

- PR-AUC (average precision, at-risk class): 0.851
- ROC-AUC: 0.892
- Brier: 0.129
- Mean predicted risk 0.392 against actual 0.393
- At threshold 0.5: precision 0.748, recall 0.711, F1 0.729, accuracy 0.793

Calibration by decile is close throughout (for example, predicted 0.316 vs actual 0.338, predicted 0.813 vs actual 0.799). Top decile: mean predicted 0.997, actual 1.000. The 0.5 threshold is for reporting only; the operating threshold should be set by revenue staff.

Logistic regression baseline: PR-AUC 0.8135, ROC-AUC 0.858.

## Compute and time

- Grid search: about 40 seconds wall clock for the whole experiment script: 5 LightGBM configs (each up to 1500 rounds, evaluated at 8 checkpoints) plus the logistic baseline.
- Final training: 2.6 seconds.
- Scoring: 0.3 seconds for 17,730 rows.

## Caveats

- Hyperparameters were chosen on validation, so the 0.851 is slightly optimistic. The grid is flat, so the effect is probably under 0.005.
- Future-period drift: base rate and mix have moved between train and validation. Monitor the mean of risk_score against observed outcomes once the test outcomes are known.
- ADR cap at 1000 is a judgement call on the March 2024 burst; a real price increase above 1000 would be scored as 1000.
- Unseen categories fall into OTHER. Unseen agents (29 in validation) are scored like the rarest trained agents.

## Reproduce

- `./.venv/bin/python deliverable/train.py` refits and saves the model (about 3 seconds).
- Experiment scripts (grid, ablations, ADR check, scoring check) are in the work folder next to `deliverable/`, not in the deliverable itself.
