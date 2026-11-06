# Hotel booking cancellation-risk model

## Model and rationale

The saved estimator is a LightGBM gradient-boosted tree classifier (500 trees, learning rate 0.035, 23 leaves, minimum 100 rows per leaf, column fraction 0.9, L1 0.2, L2 1.0, random seed 42). LightGBM was chosen because it handles nonlinear effects, missing numeric values, and interactions between booking attributes efficiently. Its native categorical splits also avoid a large sparse feature matrix. The complete saved artifact is about 0.60 MB and scoring requires one model.

The positive class is `is_canceled` represented by `1`, `true`, or `Yes`; this agrees exactly with reservation outcomes Canceled and No-Show in the supplied tables. The final model and categorical encoder were fitted only on all 83,067 rows of `train`. The 17,730 later-period `validation` rows were used only for model/feature selection and reporting.

## Features available at booking creation

Raw numeric features:

- lead time
- arrival year, week number, and day of month
- weekend and weekday stay lengths
- adults, children, and babies
- repeated-guest flag
- previous cancellations and previous non-canceled bookings
- ADR
- number of special requests

Raw categorical features:

- hotel
- meal
- country
- market segment
- distribution channel
- reserved room type
- deposit type
- agent
- customer type
- promotion code

Engineered features include numeric/cyclic arrival month, total nights, party size, child-party flag, prior booking total and prior cancellation rate, ADR per person, log ADR, log lead time, reconstructed arrival-date ordinal/day-of-week/day and week cycles, and estimated booking-date ordinal/month/day-of-week (arrival date minus lead time, with cyclic booking month).

The booking ID is returned but is not a predictor. Company was tested and removed because it did not improve future-period validation. `is_canceled`, `reservation_status`, and `reservation_status_date` are outcome leakage. `assigned_room_type` and `booking_changes` were also excluded because they can be determined or changed after the booking is created, so they are unavailable for the stated scoring moment.

## Cleaning

All model inputs are cleaned inside `predict.py`:

- Numeric text is stripped and parsed with invalid values converted to missing values, which LightGBM handles natively.
- Negative lead times and lead times above 1,000 days are flagged as invalid and clipped to the plausible range.
- Full and abbreviated month names are mapped to month numbers.
- Categorical text is trimmed, lowercased, and has repeated whitespace collapsed.
- Blank and sentinel values such as `-`, `NULL`, `NA`, `N/A`, and `unknown` become a common missing category.
- Common variants and typographical errors are canonicalized, including full country names versus ISO-like codes, `Onlin TA`, `Offline TA/T0`, `No Depost`, hyphen/spacing/case variants, and `Transiant`.
- Numeric-looking agent codes such as `9.0` are canonicalized to `9`.
- Categories are encoded with a training-only ordinal encoder. Previously unseen future categories map to an explicit unknown code.
- Missing input predictor columns are handled as missing values/categories. Extra columns are ignored.

## Data issues observed

Every SQLite column is text despite mixed semantics. The target has six textual forms (`0`, `1`, `false`, `true`, `No`, `Yes`). There are numerous capitalization, abbreviation, punctuation, and typo variants in categorical columns. In training, examples include 35 invalid lead-time strings, 82 invalid and 249 blank week numbers, and 214 invalid ADR strings. Some lead times are negative and one reaches 4,989 days. Category fields contain blanks and sentinel strings. There are 244 duplicate booking IDs in training and 56 in validation; rows are retained because the scoring contract requires one result per input row, not one per unique ID.

The outcome prevalence increases from 35.12% in training to 39.26% in the later validation period, so temporal drift is present. The validation mean predicted risk is 37.43%. No validation-based probability correction was saved, because that would make the delivered model partly fitted on validation. Reservation status identifies the outcome exactly and was deliberately excluded.

## Tuning and validation

Small, manual, deterministic comparisons were run on the supplied later-period validation set. They covered 7, 11, 15, 23, 31, and 63 leaves; roughly 450-850 boosting rounds; minimum leaf sizes of 60-150; regularization/bagging variants; date-feature ablations; inclusion/exclusion of company; categorical interaction crosses; and a one-hot XGBoost comparison. The date-derived features improved average precision; company and explicit high-cardinality crosses did not. The selected single 23-leaf model was preferred over a multi-model ensemble because the ensemble gain was only about 0.0005 average precision while doubling scoring work.

Later-period validation results for the delivered model:

- PR-AUC / average precision: 0.88153
- ROC-AUC: 0.91332
- Log loss: 0.35662
- Brier score: 0.11775
- At threshold 0.5: precision 0.80766, recall 0.72759, F1 0.76553, accuracy 0.82504

The 0.5 threshold is reported only for reference. Revenue users can choose a threshold based on overbooking capacity and the relative costs of an empty room versus walking a guest; `risk_score` itself should be used for ranking and decision rules.
