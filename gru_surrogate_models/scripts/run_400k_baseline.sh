#!/usr/bin/env bash
# Train the reference configs on the 400k set, then score every finished run and
# their averages on the test split, with a per-source (uniform/sobol) breakdown.
# Usage: [RUNS="42_event_gru_addlog:1:300"] [EVAL_ONLY=1] scripts/run_400k_baseline.sh
# Env: CSV_PATH (data/data_400k.csv), RUNS (config:seed:max_epochs triples), BOOTSTRAP (1000), EVAL_ONLY (0).
# Run ids go to logs/manifest_400k.csv, logs to logs/400k_<config>_seed<seed>.log, the table to
# logs/400k_baseline_table.log and logs/400k_baseline_preds.npz; a (config, seed) whose checkpoint exists is skipped.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

CSV_PATH=${CSV_PATH:-data/data_400k.csv}
RUNS=${RUNS:-"42_event_gru_addlog:1:300 24_gru_xl:1:200 24_gru_xl:2:200 42_event_gru_addlog:2:300 42_event_gru_addlog:3:300"}
BOOTSTRAP=${BOOTSTRAP:-1000}
EVAL_ONLY=${EVAL_ONLY:-0}
MANIFEST=logs/manifest_400k.csv
SIDECAR="${CSV_PATH}.series_map.csv"
PREDS=logs/400k_baseline_preds.npz
TABLE=logs/400k_baseline_table.log

require_env "$CSV_PATH"
manifest_init "$MANIFEST" "config,seed,max_epochs,run_id,status"

if [ "$EVAL_ONLY" != "1" ]; then
    for spec in $RUNS; do
        IFS=: read -r name seed epochs <<< "$spec"
        config=$(config_path "$name")
        [ -f "configs/$config" ] || { echo "!! configs/$config missing, skipping" >&2; continue; }
        skip_if_trained "$MANIFEST" "$config" "$seed" && continue
        train_one "logs/400k_${name}_seed${seed}.log" "$config" "$seed" "$epochs"
        manifest_upsert "$MANIFEST" "$config" "$seed" "$RUN_ID" "$STATUS" "$epochs"
    done
fi

echo
echo "==== $(date '+%F %T') final table: $CSV_PATH test split"
mapfile -t members < <(manifest_members "$MANIFEST")
n=$(( ${#members[@]} / 2 ))
[ "$n" -gt 0 ] || { echo "no finished runs in $MANIFEST yet"; exit 0; }

idx42=(); idx24=()
for ((i = 0; i < n; i++)); do
    case "${members[2 * i + 1]}" in
        *42_event_gru_addlog*) idx42+=("$i") ;;
        *24_gru_xl*) idx24+=("$i") ;;
    esac
done
groups=()
[ "${#idx42[@]}" -gt 1 ] && groups+=(--group "42 x${#idx42[@]}=$(IFS=,; echo "${idx42[*]}")")
[ "${#idx24[@]}" -gt 1 ] && groups+=(--group "24_gru_xl x${#idx24[@]}=$(IFS=,; echo "${idx24[*]}")")
[ "$n" -gt 1 ] && groups+=(--group "all $n=$(seq -s, 0 $((n - 1)))")

python eval_ensemble.py --csv_path "$CSV_PATH" "${members[@]}" "${groups[@]}" \
    --with_val --bootstrap "$BOOTSTRAP" --save_preds "$PREDS" 2>&1 | tee "$TABLE"

echo
echo "==== per-source breakdown of the same test predictions"
python - "$CSV_PATH" "$SIDECAR" "$PREDS" <<'PY' 2>&1 | tee -a "$TABLE"
import sys
import numpy as np
import pandas as pd
from cancer_nn.metrics import evaluate

csv_path, sidecar, preds_path = sys.argv[1:]
d = np.load(preds_path, allow_pickle=True)
y_true, preds, names = d["y_true"], d["preds"], list(d["names"])

# test split = series ids > int(0.9 * max_id), in ascending id order (the CSV row order the Dataset slices)
series = pd.read_csv(csv_path, usecols=["series"])["series"].to_numpy()[::21]
n = series.max()
test_ids = np.sort(series[series > int(n * 0.9)])
assert len(test_ids) == len(y_true), (len(test_ids), len(y_true))
src = pd.read_csv(sidecar).set_index("new_series").loc[test_ids, "source"].to_numpy()

def metrics(yt, yp):
    m = evaluate(yt, yp)
    return dict(mae=m["mae"], rmse=m["rmse"], mape=100 * m["mape"], correct=100 * m["percent"], err99=m["max_error_99"])

rows = []
for source in ["uniform", "sobol"]:
    m = src == source
    for name, yp in list(zip(names, preds)) + [(f"avg of all {len(preds)}", preds.mean(0))]:
        r = metrics(y_true[m], yp[m]); r.update(source=source, model=name, n=int(m.sum())); rows.append(r)
df = pd.DataFrame(rows)[["source", "model", "n", "mae", "rmse", "mape", "correct", "err99"]]
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
PY
echo "==== $(date '+%F %T') all done; manifest: $MANIFEST, table: $TABLE"
