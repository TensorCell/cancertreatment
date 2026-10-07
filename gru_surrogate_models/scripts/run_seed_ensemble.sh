#!/bin/bash
# Train extra seeds of one config and evaluate the prediction average of all
# its seeds in logs/seed_ensemble_manifest.csv with eval_ensemble.py.
# Usage: [CONFIG=cnn_lstm_multi_loss/39_gru_xl_hardpairs.yml] [SEEDS="1 2 3"] scripts/run_seed_ensemble.sh
# Env: CSV_PATH (data/data_400k.csv), MAX_EPOCHS (200), CONFIG (24_gru_xl), SEEDS ("3 4 5 6").
# Logs go to logs/seed_ensemble_<config>_seed<seed>.log; a seed whose checkpoint
# exists is skipped, so the script is resumable.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

CSV_PATH=${CSV_PATH:-data/data_400k.csv}
MAX_EPOCHS=${MAX_EPOCHS:-200}
CONFIG=$(config_path "${CONFIG:-24_gru_xl}")
SEEDS=${SEEDS:-"3 4 5 6"}
MANIFEST=logs/seed_ensemble_manifest.csv

require_env "$CSV_PATH"
manifest_init "$MANIFEST"

for seed in $SEEDS; do
    skip_if_trained "$MANIFEST" "$CONFIG" "$seed" && continue
    train_one "logs/seed_ensemble_$(basename "$CONFIG" .yml)_seed${seed}.log" "$CONFIG" "$seed" "$MAX_EPOCHS"
    manifest_upsert "$MANIFEST" "$CONFIG" "$seed" "$RUN_ID" "$STATUS"
done

echo
echo "== evaluating the seed average of $CONFIG"
mapfile -t members < <(manifest_members "$MANIFEST" "$CONFIG")
[ ${#members[@]} -gt 0 ] || { echo "no finished runs of $CONFIG in $MANIFEST"; exit 0; }
python eval_ensemble.py --csv_path "$CSV_PATH" "${members[@]}" 2>&1 | grep -vE "Warning|warn|^\s*$"
