#!/usr/bin/env bash
# ============================================================
# PolyMentor — Model Evaluation Script
# Usage: bash scripts/evaluate.sh [--model PATH] [--split test|val]
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Colours
# ------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${CYAN}[EVALUATE]${NC} $1"; }
success(){ echo -e "${GREEN}[DONE]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
MODEL_PATH=""
SPLIT="test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --split) SPLIT="$2";  shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PROCESSED="$PROJECT_ROOT/data/processed"
MODELS_SAVED="$PROJECT_ROOT/models_saved"
LOGS_DIR="$PROJECT_ROOT/experiments/logs"
RESULTS_DIR="$PROJECT_ROOT/experiments/results"

# Default model path
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODELS_SAVED/best_mentor_model.pt"
fi

# ------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------
log "Validating inputs..."

[[ -f "$MODEL_PATH" ]] || error "Model checkpoint not found at: $MODEL_PATH\nRun: bash scripts/train.sh first."

DATA_FILE="$DATA_PROCESSED/${SPLIT}.json"
[[ -f "$DATA_FILE" ]] || error "${SPLIT}.json not found. Run: bash scripts/preprocess.sh first."

python -c "import torch; import transformers; import evaluate" >/dev/null 2>&1 \
    || error "Missing dependencies. Run: pip install -r requirements.txt"

log "Model:      $MODEL_PATH"
log "Eval split: $SPLIT ($DATA_FILE)"

# ------------------------------------------------------------
# Create results directory
# ------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/eval_${SPLIT}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
mkdir -p "$LOGS_DIR"

# ------------------------------------------------------------
# Step 1 — Error Detection Metrics
# ------------------------------------------------------------
log "Step 1/3 — Evaluating error detection (classifier metrics)..."

python "$PROJECT_ROOT/src/evaluation/evaluate.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --task error_detection \
    --output "$RUN_DIR/error_detection_results.json" \
    2>&1 | tee "$LOGS_DIR/eval_${SPLIT}_${TIMESTAMP}_detection.log"

success "Error detection results → $RUN_DIR/error_detection_results.json"

# ------------------------------------------------------------
# Step 2 — Explanation Quality Metrics
# ------------------------------------------------------------
log "Step 2/3 — Evaluating explanation quality (ROUGE, BERTScore)..."

python "$PROJECT_ROOT/src/evaluation/evaluate.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --task explanation \
    --output "$RUN_DIR/explanation_results.json" \
    2>&1 | tee "$LOGS_DIR/eval_${SPLIT}_${TIMESTAMP}_explanation.log"

success "Explanation results → $RUN_DIR/explanation_results.json"

# ------------------------------------------------------------
# Step 3 — Learning Effectiveness Score
# ------------------------------------------------------------
log "Step 3/3 — Computing Learning Effectiveness Score..."

python "$PROJECT_ROOT/src/evaluation/learning_effectiveness_score.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --output "$RUN_DIR/learning_effectiveness.json" \
    2>&1 | tee "$LOGS_DIR/eval_${SPLIT}_${TIMESTAMP}_les.log"

success "Learning effectiveness → $RUN_DIR/learning_effectiveness.json"

# ------------------------------------------------------------
# Step 4 — Error Analysis
# ------------------------------------------------------------
log "Running error analysis..."

python "$PROJECT_ROOT/src/evaluation/error_analysis.py" \
    --detection-results "$RUN_DIR/error_detection_results.json" \
    --output "$RUN_DIR/error_analysis.json" \
    2>&1 | tee -a "$LOGS_DIR/eval_${SPLIT}_${TIMESTAMP}_detection.log"

success "Error analysis → $RUN_DIR/error_analysis.json"

# ------------------------------------------------------------
# Print summary to terminal
# ------------------------------------------------------------
log "Printing summary..."

python - <<EOF
import json, os

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

det = load("$RUN_DIR/error_detection_results.json")
exp = load("$RUN_DIR/explanation_results.json")
les = load("$RUN_DIR/learning_effectiveness.json")

print("")
print("=" * 60)
print("  POLYMENTOR EVALUATION SUMMARY")
print("  Split : $SPLIT")
print("  Model : $MODEL_PATH")
print("=" * 60)
print("")
print("  ERROR DETECTION")
print(f"    Accuracy : {det.get('accuracy', 'N/A')}")
print(f"    F1 Score : {det.get('f1', 'N/A')}")
print(f"    Precision: {det.get('precision', 'N/A')}")
print(f"    Recall   : {det.get('recall', 'N/A')}")
print("")
print("  EXPLANATION QUALITY")
print(f"    ROUGE-1    : {exp.get('rouge1', 'N/A')}")
print(f"    ROUGE-L    : {exp.get('rougeL', 'N/A')}")
print(f"    BERTScore  : {exp.get('bertscore_f1', 'N/A')}")
print("")
print("  LEARNING EFFECTIVENESS")
print(f"    LES Score  : {les.get('les_score', 'N/A')}")
print(f"    Hint Util  : {les.get('hint_utilisation', 'N/A')}")
print(f"    Recurrence : {les.get('error_recurrence_rate', 'N/A')}")
print("")
print("=" * 60)
EOF

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Evaluation complete.${NC}"
echo -e "${GREEN} Results saved to: $RUN_DIR${NC}"
echo -e "${GREEN} Logs saved to:    $LOGS_DIR${NC}"
echo -e "${GREEN}============================================================${NC}"
