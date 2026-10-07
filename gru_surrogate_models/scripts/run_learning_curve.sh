#!/usr/bin/env bash
# Learning curve: train CONFIG on fractions of the series (make_data_subsets.py)
# and print each run's test metrics from its W&B summary.
# Usage: [FRACTIONS="0.5"] scripts/run_learning_curve.sh
# Env: FRACTIONS ("0.25 0.5"), SEED (1), MAX_EPOCHS (200), CONFIG (24_gru_xl), CSV_PATH (data/data_400k.csv;
# subsets are <stem>_<pct>pct.csv next to it). Run ids go to logs/learning_curve_manifest.csv keyed by
# (fraction, seed), logs to logs/lc_<pct>pct_seed<seed>.log; a fraction whose checkpoint exists is skipped.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

FRACTIONS=${FRACTIONS:-"0.25 0.5"}
SEED=${SEED:-1}
MAX_EPOCHS=${MAX_EPOCHS:-200}
CONFIG=$(config_path "${CONFIG:-24_gru_xl}")
CSV_PATH=${CSV_PATH:-data/data_400k.csv}
MANIFEST=logs/learning_curve_manifest.csv

require_env "$CSV_PATH"
manifest_init "$MANIFEST" "fraction,seed,run_id,status"

python make_data_subsets.py --csv_path "$CSV_PATH" --fractions $FRACTIONS
for frac in $FRACTIONS; do
    pct=$(python -c "print(int(round($frac * 100)))")
    csv="${CSV_PATH%.csv}_${pct}pct.csv"
    [ -f "$csv" ] || { echo "!! $csv missing, skipping" >&2; continue; }
    skip_if_trained "$MANIFEST" "$frac" "$SEED" && continue
    CSV_PATH=$csv train_one "logs/lc_${pct}pct_seed${SEED}.log" "$CONFIG" "$SEED" "$MAX_EPOCHS"
    manifest_upsert "$MANIFEST" "$frac" "$SEED" "$RUN_ID" "$STATUS"
done

echo
echo "==== learning curve, test metrics on each subset's own test split"
for frac in $FRACTIONS; do
    rid=$(manifest_run_id "$MANIFEST" "$frac" "$SEED")
    [ -n "$rid" ] || { echo "WARNING: no finished run for fraction $frac seed $SEED" >&2; continue; }
    ckpt_exists "$rid" || echo "WARNING: pretrained_models/$rid.ckpt missing for fraction $frac" >&2
    summary=$(ls wandb/*-"$rid"/files/wandb-summary.json 2>/dev/null | head -1)
    [ -n "$summary" ] || { echo "WARNING: no wandb summary for run $rid (fraction $frac)" >&2; continue; }
    python - "$frac" "$rid" "$summary" <<'PY'
import json, sys
frac, rid, path = sys.argv[1:]
s = json.load(open(path))
print(f"fraction {frac}: run {rid}  test/mae {s.get('test/mae')}  test/percent {s.get('test/percent')}  best epoch {s.get('epoch')}")
PY
done
