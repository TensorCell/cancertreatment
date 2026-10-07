#!/bin/bash
# Chain of training stages for one GPU (A: more seeds of 42, B: ranking-loss configs,
# C: train on train+val), then a comparison table with bootstrap CIs and val scores.
# Usage: [STAGES="A B"] [TRAIN_ON_VAL=1] scripts/run_next_batch.sh   (STAGES="" prints only the table)
# Env: CSV_PATH (data/data_400k.csv), MAX_EPOCHS_42 (300), SEEDS_42 ("2 3"), EPOCHS_TV_24 (155),
# EPOCHS_TV_42 (197), BOOTSTRAP (1000). Stages A and B run through run_architecture_sweep.sh;
# stage C writes logs/train_on_val_manifest.csv and logs/trainval_<config>_seed1.log. Resumable.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/_lib.sh

CSV_PATH=${CSV_PATH:-data/data_400k.csv}
STAGES=${STAGES-"A B"}
TRAIN_ON_VAL=${TRAIN_ON_VAL:-0}
MAX_EPOCHS_42=${MAX_EPOCHS_42:-300}
SEEDS_42=${SEEDS_42:-"2 3"}
EPOCHS_TV_24=${EPOCHS_TV_24:-155}
EPOCHS_TV_42=${EPOCHS_TV_42:-197}
BOOTSTRAP=${BOOTSTRAP:-1000}
ARCH_MANIFEST=logs/architecture_manifest.csv
TV_MANIFEST=logs/train_on_val_manifest.csv
PREDS=logs/next_batch_preds.npz
[ "$TRAIN_ON_VAL" == "1" ] && STAGES="$STAGES C"

# The user's local checkpoints of the earlier GRU runs (config:run_id), the table's reference members.
OLD_GRU_RUNS=(
    cnn_lstm_multi_loss/24_gru_xl.yml:bo49tf86
    cnn_lstm_multi_loss/24_gru_xl.yml:8d1bprdn
    cnn_lstm_multi_loss/25_gru_xxl_single.yml:4w6vyc96
    cnn_lstm_multi_loss/25_gru_xxl_single.yml:n1sz38id
    cnn_lstm_multi_loss/26_gru_ensemble_diverse.yml:vtycs7j3
    cnn_lstm_multi_loss/26_gru_ensemble_diverse.yml:kd9irk1k
)

require_env "$CSV_PATH"

stage_a() {
    echo "==== $(date '+%F %T') stage A: 42_event_gru_addlog seeds $SEEDS_42 at $MAX_EPOCHS_42 epochs"
    CONFIGS="42_event_gru_addlog" SEEDS="$SEEDS_42" MAX_EPOCHS="$MAX_EPOCHS_42" scripts/run_architecture_sweep.sh
}

stage_b() {
    echo "==== $(date '+%F %T') stage B: ranking-loss configs 40 then 39"
    CONFIGS="40_gru_xl_margin_0 39_gru_xl_hardpairs" SEEDS="1" scripts/run_architecture_sweep.sh
}

stage_c() {
    echo "==== $(date '+%F %T') stage C: train on train+val, fixed epochs"
    manifest_init "$TV_MANIFEST"
    for spec in "24_gru_xl:$EPOCHS_TV_24" "42_event_gru_addlog:$EPOCHS_TV_42"; do
        name=${spec%%:*}; epochs=${spec##*:}; config=$(config_path "$name"); seed=1
        skip_if_trained "$TV_MANIFEST" "$config" "$seed" && continue
        train_one "logs/trainval_${name}_seed${seed}.log" "$config" "$seed" "$epochs" -train_on_val
        manifest_upsert "$TV_MANIFEST" "$config" "$seed" "$RUN_ID" "$STATUS"
    done
}

for stage in $STAGES; do
    case "$stage" in
        A) stage_a ;;
        B) stage_b ;;
        C) stage_c ;;
        *) echo "unknown stage $stage" >&2 ;;
    esac
done

echo
echo "==== $(date '+%F %T') final comparison (test + val, $BOOTSTRAP bootstrap resamples)"
members=(); names=(); tv_flags=()
add() {  # CONFIG RUN_ID
    ckpt_exists "$2" || { echo "WARNING: pretrained_models/$2.ckpt missing, skipping $1" >&2; return; }
    members+=(--member "$1:pretrained_models/$2.ckpt"); names+=("$1")
}
idx_of() { local out=(); for i in "${!names[@]}"; do [[ "${names[$i]}" == *"$1"* ]] && out+=("$i"); done; (IFS=,; echo "${out[*]}"); }

for m in "${OLD_GRU_RUNS[@]}"; do add "${m%%:*}" "${m##*:}"; done
# Group indices are taken now, before the manifests add members, so missing old checkpoints do not shift them.
ref_idx=$(idx_of 24_gru_xl)
n_old=${#names[@]}
while IFS=, read -r c _ r st; do
    [ "$st" == "OK" ] || continue
    case "$c" in *24_gru_xl*|*25_gru*|*26_gru*|*ct_transformer*) continue ;; esac
    add "$c" "$r"
done < <(tail -n +2 "$ARCH_MANIFEST" 2>/dev/null)
if [ -f "$TV_MANIFEST" ]; then
    while IFS=, read -r c _ r st; do
        [ "$st" == "OK" ] && ckpt_exists "$r" && { add "$c" "$r"; tv_flags+=(--train_on_val $((${#names[@]} - 1))); }
    done < <(tail -n +2 "$TV_MANIFEST")
fi
[ ${#names[@]} -gt 0 ] || { echo "no checkpoints to evaluate"; exit 0; }

groups=()
[ -n "$ref_idx" ] && groups+=(--group "gru_xl (ref)=$ref_idx")
[ "$n_old" -gt 0 ] && groups+=(--group "old $n_old GRU runs=$(seq -s, 0 $((n_old - 1)))")
[ -n "$(idx_of 42_event_gru_addlog)" ] && groups+=(--group "42 all seeds=$(idx_of 42_event_gru_addlog)")
groups+=(--group "all GRU-family=$(seq -s, 0 $((${#names[@]} - 1)))")
python eval_ensemble.py --csv_path "$CSV_PATH" "${members[@]}" "${tv_flags[@]}" "${groups[@]}" \
    --with_val --bootstrap "$BOOTSTRAP" --save_preds "$PREDS" 2>&1 | grep -vE "Warning|warn|^\s*$"
echo "predictions saved to $PREDS (test + val, per member)"
