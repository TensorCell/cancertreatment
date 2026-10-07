# Shared helpers for the runner scripts in this directory. Source it after
# cd'ing to surrogate_models; every path below is relative to that directory.
# A manifest is a CSV keyed by (config, seed) with columns
# config,seed[,max_epochs],run_id,status; status is OK or FAILED.

require_env() {  # CSV_PATH
    python -c "import pytorch_lightning" 2>/dev/null || {
        echo "ERROR: pytorch_lightning not importable, activate the training env first (conda activate cancer-nn-5070)" >&2
        exit 1
    }
    [ -n "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE:-}" = offline ] || grep -qs api.wandb.ai ~/.netrc || {
        echo "ERROR: no W&B credentials, export WANDB_API_KEY or run 'wandb login'" >&2
        exit 1
    }
    [ -f "$1" ] || { echo "ERROR: $1 not found" >&2; exit 1; }
    mkdir -p logs pretrained_models
}

config_path() {  # NAME or cnn_lstm_multi_loss/NAME.yml -> cnn_lstm_multi_loss/NAME.yml
    case "$1" in
        */*) echo "$1" ;;
        *) echo "cnn_lstm_multi_loss/${1%.yml}.yml" ;;
    esac
}

manifest_init() {  # MANIFEST [HEADER]
    [ -f "$1" ] || echo "${2:-config,seed,run_id,status}" > "$1"
}

manifest_upsert() {  # MANIFEST KEY SEED RUN_ID STATUS [EXTRA]  (EXTRA is the third column when given)
    local manifest=$1 key=$2 seed=$3 run_id=$4 status=$5 tmp row
    row="$key,$seed,${6:+$6,}$run_id,$status"
    tmp=$(mktemp)
    awk -F, -v k="$key" -v s="$seed" 'NR == 1 || !($1 == k && $2 == s)' "$manifest" > "$tmp"
    echo "$row" >> "$tmp"
    mv "$tmp" "$manifest"
}

manifest_run_id() {  # MANIFEST KEY SEED -> run id of the OK row, if any
    awk -F, -v k="$2" -v s="$3" '
        NR == 1 { for (i = 1; i <= NF; i++) { if ($i == "run_id") r = i; if ($i == "status") st = i }; next }
        $1 == k && $2 == s && $st == "OK" { print $r }' "$1"
}

ckpt_exists() {  # RUN_ID
    [ -f "pretrained_models/$1.ckpt" ]
}

skip_if_trained() {  # MANIFEST KEY SEED -> true (and a message) when an OK run with a checkpoint exists
    local run_id
    run_id=$(manifest_run_id "$1" "$2" "$3")
    if [ -z "$run_id" ] || ! ckpt_exists "$run_id"; then
        return 1
    fi
    echo "== $2 seed $3 already trained ($run_id), skipping"
}

wandb_runs() {
    ls wandb/ 2>/dev/null | sort
}

new_wandb_run_id() {  # BEFORE (output of wandb_runs taken before the run) -> id of the run dir added since
    local new_dir
    new_dir=$(comm -13 <(printf '%s\n' "$1") <(wandb_runs) | grep -E '^(offline-)?run-' | head -1)
    echo "${new_dir##*-}"
}

# Runs main.py on $CSV_PATH, teeing to LOG; sets RUN_ID and STATUS (OK/FAILED) and returns 0 on OK.
train_one() {  # LOG CONFIG SEED MAX_EPOCHS [extra main.py args...]
    local log=$1 config=$2 seed=$3 max_epochs=$4 before rc
    shift 4
    echo "== $(date '+%F %T') training $config seed $seed, max $max_epochs epochs, on $CSV_PATH -> $log"
    before=$(wandb_runs)
    python main.py -csv_path "$CSV_PATH" -config "$config" -config_folder configs \
        -wandb_config wandb_config.yml -saved_model_path pretrained_models \
        -max_epochs "$max_epochs" -seed "$seed" "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    RUN_ID=$(new_wandb_run_id "$before")
    if [ "$rc" -eq 0 ] && [ -n "$RUN_ID" ]; then
        STATUS=OK
        echo "== $(date '+%F %T') done: $config seed $seed -> run $RUN_ID"
    else
        STATUS=FAILED
        echo "!! $config seed $seed failed (exit $rc), see $log" >&2
        return 1
    fi
}

# Prints the eval_ensemble.py --member args, one per line, for every OK row of
# MANIFEST (only KEY's rows when given) whose checkpoint is on disk.
manifest_members() {  # MANIFEST [KEY]
    local c r
    while read -r c r; do
        ckpt_exists "$r" && printf -- '--member\n%s:pretrained_models/%s.ckpt\n' "$c" "$r"
    done < <(awk -F, -v k="${2:-}" '
        NR == 1 { for (i = 1; i <= NF; i++) { if ($i == "run_id") r = i; if ($i == "status") st = i }; next }
        (k == "" || $1 == k) && $st == "OK" { print $1, $r }' "$1")
}
