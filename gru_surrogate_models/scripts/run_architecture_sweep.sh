#!/bin/bash
# Train the architecture candidates on several seeds, then print each config's
# test metrics per seed and seed-averaged next to the 24_gru_xl reference.
# Usage: [CONFIGS="41_event_gru 44_ct_transformer"] [SEEDS="1 2"] [EVAL_ONLY=1] scripts/run_architecture_sweep.sh
# Env: CSV_PATH (data/data_400k.csv), MAX_EPOCHS (200), SEEDS ("1"), CONFIGS (41..45), EVAL_ONLY (0).
# Run ids go to logs/architecture_manifest.csv, logs to logs/arch_<config>_seed<seed>.log;
# a (config, seed) whose checkpoint exists is skipped, so the script is resumable.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

CSV_PATH=${CSV_PATH:-data/data_400k.csv}
MAX_EPOCHS=${MAX_EPOCHS:-200}
SEEDS=${SEEDS:-"1"}
CONFIGS=${CONFIGS:-"41_event_gru 42_event_gru_addlog 43_gru_xl_addlog 44_ct_transformer 45_ct_transformer_addlog"}
EVAL_ONLY=${EVAL_ONLY:-0}
MANIFEST=logs/architecture_manifest.csv
REFERENCE=cnn_lstm_multi_loss/24_gru_xl.yml

require_env "$CSV_PATH"
manifest_init "$MANIFEST"

if [ "$EVAL_ONLY" != "1" ]; then
    for name in $CONFIGS; do
        config=$(config_path "$name")
        [ -f "configs/$config" ] || { echo "!! configs/$config missing, skipping" >&2; continue; }
        for seed in $SEEDS; do
            skip_if_trained "$MANIFEST" "$config" "$seed" && continue
            train_one "logs/arch_${name}_seed${seed}.log" "$config" "$seed" "$MAX_EPOCHS"
            manifest_upsert "$MANIFEST" "$config" "$seed" "$RUN_ID" "$STATUS"
        done
    done
fi

echo
echo "== comparison on the test split (per seed, then seed average per config)"
for name in "$REFERENCE" $CONFIGS; do
    config=$(config_path "$name")
    mapfile -t members < <(manifest_members "$MANIFEST" "$config")
    [ ${#members[@]} -eq 0 ] && { echo "-- $config: no finished runs"; continue; }
    echo "-- $config"
    python eval_ensemble.py --csv_path "$CSV_PATH" "${members[@]}" 2>&1 | grep -vE "Warning|warn|^\s*$"
done
