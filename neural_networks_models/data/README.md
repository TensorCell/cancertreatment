# Data: EMT6/Ro Radiotherapy Simulation Dataset 

## Source

Simulation outputs for **200,000 radiotherapy protocols** applied to the EMT6/Ro
tumour cell line: https://github.com/banasraf/EMT6-Ro. Each protocol spans 10 days: 5 days of fractionated radiotherapy
dosing, then 5 days of observation.

Dataset: https://raw.githubusercontent.com/mkmkl93/ml-ca/master/data/uniform_200k/dataset1_200.csv


## Target

**Average number of tumour cells** remaining after 10 days (already normalised).

## Constraints on dose schedules

- Maximum single dose: 2.5 Gy
- Maximum cumulative dose: 10 Gy
- Minimum inter-dose gap: > 600 seconds

## CSV Schema (`data.csv`)

| Column | Type | Description |
|---|---|---|
| (row index) | int | Row counter within the protocol |
| `time` | float | Absolute dose time in seconds; 0 for padded rows |
| `dose` | float | Single dose magnitude (≤ 2.5); 0 for padded rows |
| `series` | int | Protocol ID (0–199,999) |
| `time_idx` | int | Step index within the protocol (0–20, inclusive) |
| `is_target` | int | 1 only at step 20 (the output row) |
| `target` | float | Normalised tumour cell count; 0 for non-target rows |
| `time_gap` | float | Seconds since the last dose; 0 or negative on last row |

## Sequence structure

Every protocol has exactly **21 rows** (`time_idx` 0–20):

- **Steps 0–19** (`is_target == 0`): the radiation dose schedule, zero-padded at
  the beginning for protocols shorter than 20 doses.
- **Step 20** (`is_target == 1`): the outcome row.  Only `target` is meaningful
  here; `time`, `dose`, and `time_gap` are set to 0.

## Pre-processing applied by `cancer_dataset.py`

1. Load the CSV and group by `series`.
2. Extract the **dose schedule** (steps 0–19) as `past_target` (shape `(20,)`).
3. Extract `time` and `time_gap` for steps 0–19 as **historical covariates** (shape `(20, 2)`).
4. Extract `target` at step 20 as the **label** (shape `(1,)`).
5. Apply **MinMaxScaler** (fit on training split only) to `dose`, `time`, and `time_gap`.
6. `target` is already normalised and is used as-is.
7. Split 70 / 15 / 15 by protocol ID (shuffled with a fixed seed).
