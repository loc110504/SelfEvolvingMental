#!/usr/bin/env bash
# PsyVEC driver: one config block, one backend switch, and the stages that make
# up an evolution round.
#
#   ./scripts/run_psyvec.sh <stage> [extra args passed through]
#
# Stages
#   fit       Fit the imputation model (and a calibrator, once a train run
#             exists) on the TRAINING split only.
#   train     Assess the training split. Produces the states the lesson round
#             mines, so it must run with evidence caching (the default).
#   evolve    Mine, distil and transfer-verify scorer lessons from the train run.
#   dev       Assess the development split.
#   test      Assess the held-out test split. Run this LAST and once.
#   score     Print metrics for a finished run directory.
#   ablate    Run the development split through every ablation arm.
#   all       fit -> train -> evolve -> dev -> score
#
# Lessons are off until `evolve` has produced some; turn them on with
#   LESSONS=configs/fitted/lessons.json ./scripts/run_psyvec.sh dev
#
# Backend is chosen by MODE:
#   openai   api.openai.com                      (needs OPENAI_API_KEY)
#   vllm     any OpenAI-compatible server        (needs BASE_URL)
#   local    in-process transformers             (needs a GPU for a large model)
#
# Everything below can be overridden from the environment, e.g.
#   MODE=vllm MODEL=Qwen/Qwen2.5-14B-Instruct ./scripts/run_psyvec.sh dev

set -euo pipefail

# ─────────────────────────── configuration ────────────────────────────
MODE="${MODE:-openai}"

case "$MODE" in
  openai) MODEL="${MODEL:-gpt-4o-mini}";              BASE_URL="${BASE_URL:-}"                          ; CONCURRENCY="${CONCURRENCY:-8}"  ;;
  vllm)   MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"; BASE_URL="${BASE_URL:-http://localhost:8000/v1}"  ; CONCURRENCY="${CONCURRENCY:-16}" ;;
  local)  MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}";  BASE_URL=""                                       ; CONCURRENCY="${CONCURRENCY:-1}"  ;;
  *) echo "MODE must be one of: openai vllm local (got '$MODE')" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-$(echo "$MODE-$MODEL" | tr '/:' '__' | tr '[:upper:]' '[:lower:]')}"
DATA_ROOT="${DATA_ROOT:-data/processed_daic_woz}"
OUT_ROOT="${OUT_ROOT:-results/evaluations/$RUN_TAG}"
FITTED="${FITTED:-configs/fitted}"

MAX_ROUNDS="${MAX_ROUNDS:-2}"
EVIDENCE_LIMIT="${EVIDENCE_LIMIT:-8}"
DEVICE="${DEVICE:-}"
# Imputer: prefer the variant fitted on an actual training run, which absorbs
# the scorer's systematic offset, and fall back to the label-only one so the
# pipeline is runnable before any train run exists.
if [ -z "${IMPUTER:-}" ]; then
  if [ -f "$FITTED/imputation_run.json" ]; then
    IMPUTER="$FITTED/imputation_run.json"
  else
    IMPUTER="$FITTED/imputation_references.json"
  fi
fi
CALIBRATOR="${CALIBRATOR:-}"
LESSONS="${LESSONS:-}"
# gpt-4o-mini list pricing at time of writing; re-check before quoting a cost.
IN_COST="${IN_COST:-0.15}"
OUT_COST="${OUT_COST:-0.60}"

PY="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

# ─────────────────────────── shared arguments ─────────────────────────
backend_args() {
  printf -- '--backend %s --model %s --concurrency %s' "$MODE" "$MODEL" "$CONCURRENCY"
  [ -n "$BASE_URL" ] && printf -- ' --base-url %s' "$BASE_URL"
  [ -n "$DEVICE" ]   && printf -- ' --device %s' "$DEVICE"
  printf -- ' --input-cost-per-1m %s --output-cost-per-1m %s' "$IN_COST" "$OUT_COST"
}

pipeline_args() {
  printf -- '--max-rounds %s --evidence-limit %s' "$MAX_ROUNDS" "$EVIDENCE_LIMIT"
  [ -f "$IMPUTER" ]     && printf -- ' --imputation-model %s' "$IMPUTER"
  [ -n "$CALIBRATOR" ] && [ -f "$CALIBRATOR" ] && printf -- ' --calibrator %s' "$CALIBRATOR"
  [ -n "$LESSONS" ]    && [ -f "$LESSONS" ]    && printf -- ' --lessons %s' "$LESSONS"
}

assess() {  # assess <split> [extra args...]
  local split="$1"; shift
  echo "── assess ${split} | mode=$MODE model=$MODEL ─────────────────────"
  # shellcheck disable=SC2046  # word splitting of the arg builders is intended
  $PY scripts/run_assessment.py \
      --data-dir "$DATA_ROOT/$split" \
      --output-dir "$OUT_ROOT/$split" \
      $(backend_args) $(pipeline_args) --skip-existing "$@"
}

score() {  # score <split> [run dir]
  local split="$1"; local dir="${2:-$OUT_ROOT/$split}"
  $PY scripts/score_run.py "$dir" --split "$split"
}

require_key() {
  if [ "$MODE" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is not set. Export it, or run with MODE=vllm." >&2
    exit 2
  fi
}

# ─────────────────────────────── stages ───────────────────────────────
STAGE="${1:-all}"; shift || true

case "$STAGE" in
  fit)
    if [ -d "$OUT_ROOT/train" ] && compgen -G "$OUT_ROOT/train/*_evaluation.json" >/dev/null; then
      echo "── fitting imputer + calibrator from the train RUN ──"
      $PY scripts/fit_psyvec_priors.py --from-run "$OUT_ROOT/train" --out-dir "$FITTED" "$@"
    else
      echo "── no train run yet: fitting the label-only imputer ──"
      $PY scripts/fit_psyvec_priors.py --out-dir "$FITTED" "$@"
    fi
    ;;

  train) require_key; assess train "$@" ;;
  dev)   require_key; assess dev   "$@" ;;
  test)  require_key; assess test  "$@" ;;

  evolve)
    require_key
    echo "── mining, distilling and transfer-verifying lessons ──"
    # shellcheck disable=SC2046
    $PY scripts/evolve_lessons.py \
        --train-run "$OUT_ROOT/train" \
        --backend "$MODE" --model "$MODEL" --concurrency "$CONCURRENCY" \
        $([ -n "$BASE_URL" ] && printf -- '--base-url %s' "$BASE_URL") \
        $([ -f "$IMPUTER" ] && printf -- '--imputation-model %s' "$IMPUTER") \
        --out "$FITTED/lessons.json" "$@"
    echo "Re-run a split with LESSONS=$FITTED/lessons.json to use them."
    ;;

  score)
    score "${1:-dev}" "${2:-}"
    ;;

  ablate)
    require_key
    echo "── ablation sweep on dev ──"
    # Each arm names the ONE thing it removes. An arm that changes a value the
    # config already sets overrides the variable rather than appending a second
    # copy of the flag, so the command stays readable in the log.
    for arm in "full::removes nothing" \
               "no-imputation:--no-imputation:abstentions score 0, the v2 rule" \
               "no-expansion::1" \
               "no-polarity:--no-polarity:reverse-polarity excerpts unlabelled" \
               "no-turn-merge:--no-turn-merge:split answers left fragmented"; do
      name="${arm%%:*}"; rest="${arm#*:}"
      flags="${rest%%:*}"; note="${rest#*:}"
      rounds=""; [ "$name" = "no-expansion" ] && { rounds=1; note="one retrieval round only"; }
      echo; echo "══ arm: $name — $note ══"
      # shellcheck disable=SC2046,SC2086
      $PY scripts/run_assessment.py \
          --data-dir "$DATA_ROOT/dev" \
          --output-dir "$OUT_ROOT/ablate_$name" \
          $(backend_args) \
          --max-rounds "${rounds:-$MAX_ROUNDS}" --evidence-limit "$EVIDENCE_LIMIT" \
          $([ -f "$IMPUTER" ] && printf -- '--imputation-model %s' "$IMPUTER") \
          $([ -n "$LESSONS" ] && [ -f "$LESSONS" ] && printf -- '--lessons %s' "$LESSONS") \
          --skip-existing $flags
      score dev "$OUT_ROOT/ablate_$name"
    done
    ;;

  all)
    require_key
    "$0" fit
    "$0" train
    "$0" evolve
    LESSONS="$FITTED/lessons.json" "$0" dev
    score dev
    ;;

  *)
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
    exit 2
    ;;
esac
