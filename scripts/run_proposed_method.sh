#!/usr/bin/env bash
# The proposed method, end to end: bounded self-evolving evidence memory for
# adaptive PHQ-8 interviewing, on DAIC-WOZ.
#
#   ./scripts/run_proposed_method.sh <stage> [extra args passed through]
#
# Stages
#   fit       Fit the imputation model (and a calibrator, once a train run
#             exists) on the TRAINING split only.
#   train     Run the adaptive-interview pipeline over the training split,
#             building each participant's Patient Evidence Memory and
#             caching the acquisition traces the evolve stage mines.
#   evolve    Mine, distil and transfer-verify BOTH bounded-evolution
#             channels from the train run only: scorer-interpretation
#             lessons (scripts/evolve_lessons.py) and acquisition-strategy
#             lessons (scripts/evolve_interview_lessons.py).
#   freeze    No computation — a checkpoint stage that only confirms the
#             lesson files evolve just wrote exist, so every stage after it
#             is documented as evaluate-only.
#   dev       Assess the development split with both lesson files frozen.
#   test      Assess the held-out test split, same frozen memory. Run this
#             LAST and once.
#   score     Print metrics and build the Evidence-Criterion Map artifact
#             for one split's run directory.
#   all       fit -> train -> evolve -> freeze -> dev -> test -> score(dev)
#             -> score(test)
#
# No ablation arms: this runs only the full proposed method.
#
# Backend is chosen by MODE (default: ollama):
#   ollama   a local Ollama server's OpenAI-compatible endpoint, model
#            gpt-oss:20b by default        (needs `ollama serve` + the model
#                                            pulled: `ollama pull gpt-oss:20b`)
#   openai   api.openai.com, model gpt-4o-mini by default
#                                           (needs OPENAI_API_KEY)
#   vllm     any other OpenAI-compatible server (needs BASE_URL)
#   local    in-process transformers        (needs a GPU for a large model)
#
# Everything below can be overridden from the environment, e.g.
#   MODE=openai MODEL=gpt-4o-mini ./scripts/run_proposed_method.sh all

set -euo pipefail

# ─────────────────────────── configuration ────────────────────────────
MODE="${MODE:-ollama}"

case "$MODE" in
  ollama) MODEL="${MODEL:-gpt-oss:20b}";               BASE_URL="${BASE_URL:-http://localhost:11434/v1}"; CONCURRENCY="${CONCURRENCY:-4}"  ;;
  openai) MODEL="${MODEL:-gpt-4o-mini}";               BASE_URL="${BASE_URL:-}"                          ; CONCURRENCY="${CONCURRENCY:-8}"  ;;
  vllm)   MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"; BASE_URL="${BASE_URL:-http://localhost:8000/v1}"  ; CONCURRENCY="${CONCURRENCY:-16}" ;;
  local)  MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}";  BASE_URL=""                                       ; CONCURRENCY="${CONCURRENCY:-1}"  ;;
  *) echo "MODE must be one of: ollama openai vllm local (got '$MODE')" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-$(echo "proposed-$MODE-$MODEL" | tr '/:' '__' | tr '[:upper:]' '[:lower:]')}"
DATA_ROOT="${DATA_ROOT:-data/processed_daic_woz}"
OUT_ROOT="${OUT_ROOT:-results/proposed/$RUN_TAG}"
FITTED="${FITTED:-configs/fitted}"

GLOBAL_BUDGET="${GLOBAL_BUDGET:-6}"
MAX_ROUNDS_PER_TOPIC="${MAX_ROUNDS_PER_TOPIC:-3}"
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
SCORER_LESSONS="${SCORER_LESSONS:-$FITTED/lessons.json}"
INTERVIEW_LESSONS="${INTERVIEW_LESSONS:-$FITTED/interview_lessons.json}"
# gpt-4o-mini list pricing at time of writing; re-check before quoting a cost.
# Zero for a local Ollama model, which has no per-token API cost.
if [ "$MODE" = "ollama" ] || [ "$MODE" = "vllm" ] || [ "$MODE" = "local" ]; then
  IN_COST="${IN_COST:-0.0}"
  OUT_COST="${OUT_COST:-0.0}"
else
  IN_COST="${IN_COST:-0.15}"
  OUT_COST="${OUT_COST:-0.60}"
fi

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
  printf -- '--global-budget %s --max-rounds-per-topic %s --evidence-limit %s' \
    "$GLOBAL_BUDGET" "$MAX_ROUNDS_PER_TOPIC" "$EVIDENCE_LIMIT"
  [ -f "$IMPUTER" ]             && printf -- ' --imputation-model %s' "$IMPUTER"
  [ -n "$CALIBRATOR" ] && [ -f "$CALIBRATOR" ]         && printf -- ' --calibrator %s' "$CALIBRATOR"
}

frozen_lesson_args() {
  [ -f "$SCORER_LESSONS" ]    && printf -- ' --lessons %s' "$SCORER_LESSONS"
  [ -f "$INTERVIEW_LESSONS" ] && printf -- ' --interview-lessons %s' "$INTERVIEW_LESSONS"
}

assess() {  # assess <split> [extra args...]
  local split="$1"; shift
  echo "── assess ${split} | mode=$MODE model=$MODEL ─────────────────────"
  # shellcheck disable=SC2046  # word splitting of the arg builders is intended
  $PY scripts/run_proposed_assessment.py \
      --data-dir "$DATA_ROOT/$split" \
      --output-dir "$OUT_ROOT/$split" \
      $(backend_args) $(pipeline_args) "$@" --skip-existing
}

score() {  # score <split> [run dir]
  local split="$1"; local dir="${2:-$OUT_ROOT/$split}"
  $PY scripts/score_run.py "$dir" --split "$split"
  $PY scripts/build_evidence_criterion_map.py "$dir" \
      --markdown "$dir/criterion_maps.md"
}

require_key() {
  case "$MODE" in
    openai)
      if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "OPENAI_API_KEY is not set. Export it, or run with MODE=ollama." >&2
        exit 2
      fi
      ;;
    ollama)
      local root="${BASE_URL%/v1}"
      if ! curl -sf "$root/api/tags" >/dev/null 2>&1; then
        echo "Cannot reach Ollama at $root — is 'ollama serve' running?" >&2
        exit 2
      fi
      if ! curl -sf "$root/api/tags" | grep -q "$MODEL"; then
        echo "Ollama is running but '$MODEL' is not pulled. Run: ollama pull $MODEL" >&2
        exit 2
      fi
      ;;
  esac
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

  train)
    require_key
    # No frozen lesson files at this stage: this run is what evolve mines from.
    assess train "$@"
    ;;

  evolve)
    require_key
    echo "── channel 1/2: mining scorer-interpretation lessons ──"
    # shellcheck disable=SC2046
    $PY scripts/evolve_lessons.py \
        --train-run "$OUT_ROOT/train" --train-dir "$DATA_ROOT/train" \
        --backend "$MODE" --model "$MODEL" --concurrency "$CONCURRENCY" \
        $([ -n "$BASE_URL" ] && printf -- '--base-url %s' "$BASE_URL") \
        $([ -f "$IMPUTER" ] && printf -- '--imputation-model %s' "$IMPUTER") \
        --out "$SCORER_LESSONS"
    echo "── channel 2/2: mining acquisition-strategy lessons ──"
    # shellcheck disable=SC2046
    $PY scripts/evolve_interview_lessons.py \
        --train-run "$OUT_ROOT/train" --train-dir "$DATA_ROOT/train" \
        --backend "$MODE" --model "$MODEL" --concurrency "$CONCURRENCY" \
        --evidence-limit "$EVIDENCE_LIMIT" \
        $([ -n "$BASE_URL" ] && printf -- '--base-url %s' "$BASE_URL") \
        --out "$INTERVIEW_LESSONS"
    ;;

  freeze)
    echo "── freeze: from here on, lesson files are read-only ──"
    for f in "$SCORER_LESSONS" "$INTERVIEW_LESSONS"; do
      if [ -f "$f" ]; then
        echo "  frozen: $f"
      else
        echo "  (none promoted: $f does not exist — dev/test will run without this channel)"
      fi
    done
    ;;

  dev)  require_key; assess dev  $(frozen_lesson_args) "$@" ;;
  test) require_key; assess test $(frozen_lesson_args) "$@" ;;

  score)
    score "${1:-dev}" "${2:-}"
    ;;

  all)
    require_key
    # Invoked as "bash $0 ..." rather than executed directly: the checkout's
    # executable bit is not guaranteed (e.g. on a mounted/network filesystem
    # that does not preserve Unix permissions), so re-invoking bare "$0" here
    # would fail with "Permission denied" even though `bash "$0"` works fine.
    bash "$0" fit
    bash "$0" train
    bash "$0" evolve
    bash "$0" fit
    bash "$0" freeze
    bash "$0" dev
    bash "$0" test
    score dev
    score test
    ;;

  *)
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
    exit 2
    ;;
esac
