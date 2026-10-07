#!/bin/bash
# Train each config on each seed and record the W&B run ids in logs/<manifest_name>_manifest.csv.
# Usage: scripts/train_configs.sh <manifest_name> <config> [<config>...]
#   e.g. SEEDS="1 2" scripts/train_configs.sh gru_configs 19_gru_single cnn_lstm_multi_loss/23_gru_large.yml
# Env: SEEDS ("1"), MAX_EPOCHS (200), CSV_PATH (data/data_400k.csv), EXTRA_ARGS ("", e.g. "-swa_lrs 0.0005"),
# LOG_PREFIX (<manifest_name>); logs go to logs/<LOG_PREFIX>_<config>_seed<seed>.log.
# A (config, seed) that is run again replaces its manifest row.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

[ "$#" -ge 2 ] || { echo "usage: $0 <manifest_name> <config> [<config>...]" >&2; exit 2; }
NAME=$1; shift
CSV_PATH=${CSV_PATH:-data/data_400k.csv}
MAX_EPOCHS=${MAX_EPOCHS:-200}
SEEDS=${SEEDS:-1}
LOG_PREFIX=${LOG_PREFIX:-$NAME}
MANIFEST=logs/${NAME}_manifest.csv
read -ra EXTRA <<< "${EXTRA_ARGS:-}"

CONFIGS=()
for c in "$@"; do
    c=$(config_path "$c")
    [ -f "configs/$c" ] || { echo "ERROR: configs/$c not found" >&2; exit 1; }
    CONFIGS+=("$c")
done

require_env "$CSV_PATH"
manifest_init "$MANIFEST"
echo "== configs: ${CONFIGS[*]} | seeds: $SEEDS | max_epochs=$MAX_EPOCHS | manifest: $MANIFEST"

for config in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
        log="logs/${LOG_PREFIX}_$(basename "$config" .yml)_seed${seed}.log"
        train_one "$log" "$config" "$seed" "$MAX_EPOCHS" "${EXTRA[@]}"
        manifest_upsert "$MANIFEST" "$config" "$seed" "$RUN_ID" "$STATUS"
    done
done

echo "== done, manifest: $MANIFEST"
