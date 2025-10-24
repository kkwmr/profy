#!/usr/bin/env bash
set -euo pipefail

# Run expert annotation alignment analysis end-to-end.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

EXPERIMENT_DIR=${1:-"$ROOT_DIR/results/experiment_20251009_224938"}
WEB_EVALS_DIR=${2:-"$ROOT_DIR/../piano-performance-marker/web_evaluations"}
AUDIO_DIR=${3:-"$ROOT_DIR/../piano-performance-marker/public/audio"}
MANIFEST=${4:-"$ROOT_DIR/../piano-performance-marker/public/audio/manifest_top20.json"}
DATA_ROOT=${5:-"$ROOT_DIR/data"}

echo "[info] Experiment dir : $EXPERIMENT_DIR"
echo "[info] Web evals dir  : $WEB_EVALS_DIR"
echo "[info] Audio dir      : $AUDIO_DIR"
echo "[info] Manifest       : $MANIFEST"
echo "[info] Data root      : $DATA_ROOT"

python -m src.evaluation.expert_alignment \
  --experiment "$EXPERIMENT_DIR" \
  --web-evals "$WEB_EVALS_DIR" \
  --audio-dir "$AUDIO_DIR" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --tune \
  --primary-metric ap \
  --severity-type product,evidence,attention \
  --severity-power 1.0,1.5,2.0 \
  --smooth-win 1,5,9 \
  --norm-minmax 0,1 \
  --lag-sec=-0.2,0.0,0.2 \
  --ann-margin-sec 0.0,0.25,0.5 \
  "$@"

echo "[done] Alignment analysis complete. See results/ for outputs."

ALIGN_DIR="$(ls -td results/expert_alignment_* 2>/dev/null | head -n1)"
if [ -n "$ALIGN_DIR" ]; then
  echo "[info] Generating sensor overlays into: $ALIGN_DIR"
  python -m src.evaluation.sensor_attention_overlays \
    --experiment "$EXPERIMENT_DIR" \
    --web-evals "$WEB_EVALS_DIR" \
    --audio-dir "$AUDIO_DIR" \
    --manifest "$MANIFEST" \
    --out "$(basename "$ALIGN_DIR")"
fi
