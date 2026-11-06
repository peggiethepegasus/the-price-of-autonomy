# Hotel cancellation / no-show risk model

## Model and target

The positive class is `reservation_status` equal to `Canceled` or `No-Show`; `Check-Out` is negative. I used LightGBM gradient-boosted decision trees (`LGBMClassifier`) because it handles nonlinear interactions and mixed numeric/categorical booking data efficiently, which is useful for effects such as deposit type interacting with lead time or market segment. The model outputs the positive-class probability as `risk_score`.

The final fit uses only the `train` table. The `validation` table was used as a later-period holdout to select the model configuration and boosting round count.

## Features and cleaning

The prediction code in `predict.py` reads the `features` table and performs the same transformations used during training. It returns the original `booking_id` in input order and one score per row.

Numeric booking fields were parsed with `to_numeric(errors="coerce")`, after stripping whitespace and treating blank, `NULL`, `None`, and `nan` as missing. The numeric fields were lead time, arrival year/week/day, weekend and weekday nights, adults, children, babies, repeated-guest flag, previous cancellations, previous non-cancelled bookings, booking changes, ADR, and special requests. LightGBM handles resulting missing numeric values.

Categorical fields were stripped and blank values were mapped to an explicit missing category. They were encoded using category vocabularies learned from train plus validation (unknown production categories become the LightGBM missing code): hotel, arrival month, meal, country, market segment, distribution channel, reserved room type, assigned room type, deposit type, agent, company, customer type, and promo code.

Additional features were derived from information available at booking creation: numeric arrival month, total nights, party size, and whether assigned and reserved room types differ. `booking_id` was not used as a predictor. `is_canceled`, `reservation_status`, and `reservation_status_date` were excluded because they contain the outcome or post-outcome information and would not be available at creation time.

## Tuning and validation

I compared four small LightGBM configurations on the chronological validation period, using average precision for model selection and early stopping (patience 80, maximum 1,500 trees). All runs used random seed 42, 0.9 row/column subsampling, and L2 regularization of 2. The selected configuration was:

- 63 leaves
- minimum child samples 80
- learning rate 0.04
- 117 boosting rounds (selected by validation early stopping)

The final model was then refit on all 83,067 training rows with that fixed configuration and 117 rounds. No validation rows were used in the final fit.

Validation results (17,730 rows):

- Average precision / PR-AUC: 0.878580
- ROC-AUC: 0.911751
- Precision at score 0.5: 0.8447
- Recall at score 0.5: 0.6680
- F1 at score 0.5: 0.7460

The at-risk rate was 0.3512 in train and 0.3926 in validation, so the later period has a noticeable rate shift. The supplied `is_canceled` field used inconsistent text encodings (`0`/`1`, `true`/`false`, `Yes`/`No`); it was deliberately not used as the target or a feature. The outcome status was consistently usable for defining the target.

## Files

- `predict.py`: self-contained scoring entry point exposing `predict_proba(db_path)`.
- `model.joblib`: saved LightGBM model and preprocessing metadata.
- `metrics.json`: recorded validation metrics and fit metadata.
