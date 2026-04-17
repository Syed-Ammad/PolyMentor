#!/usr/bin/env bash
# ============================================================
# PolyMentor — Model Training Script
# Usage: bash scripts/train.sh [--exp EXP_NAME] [--resume]
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

log()    { echo -e "${CYAN}[TRAIN]${NC} $1"; }
success(){ echo -e "${GREEN}[DONE]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
EXP_NAME="exp_$(date +%Y%m%d_%H%M%S)"
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)      EXP_NAME="$2"; shift 2 ;;
        --resume)   RESUME=true; shift ;;
        *)          error "Unknown argument: $1" ;;
    esac
done

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_MODEL="$PROJECT_ROOT/configs/model_config.yaml"
CONFIG_TRAIN="$PROJECT_ROOT/configs/training_config.yaml"
DATA_PROCESSED="$PROJECT_ROOT/data/processed"
MODELS_SAVED="$PROJECT_ROOT/models_saved"
EXP_DIR="$PROJECT_ROOT/experiments/$EXP_NAME"
LOGS_DIR="$PROJECT_ROOT/experiments/logs"

# ------------------------------------------------------------
# Validate environment
# ------------------------------------------------------------
log "Validating environment..."

python --version >/dev/null 2>&1 || error "Python not found."
python -c "import torch" >/dev/null 2>&1 || error "PyTorch not installed. Run: pip install -r requirements.txt"
python -c "import transformers" >/dev/null 2>&1 || error "Transformers not installed. Run: pip install -r requirements.txt"

# Check processed data exists
[[ -f "$DATA_PROCESSED/train.json" ]] || error "train.json not found. Run: bash scripts/preprocess.sh first."
[[ -f "$DATA_PROCESSED/val.json"   ]] || error "val.json not found. Run: bash scripts/preprocess.sh first."

# Check configs exist
[[ -f "$CONFIG_MODEL" ]] || error "model_config.yaml not found at $CONFIG_MODEL"
[[ -f "$CONFIG_TRAIN" ]] || error "training_config.yaml not found at $CONFIG_TRAIN"

# ------------------------------------------------------------
# Device detection
# ------------------------------------------------------------
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")
log "Training device: $DEVICE"

if [[ "$DEVICE" == "cpu" ]]; then
    warn "No GPU detected. Training on CPU will be slow."
    warn "Expected time: 20–60 minutes depending on dataset size."
fi

# ------------------------------------------------------------
# Create experiment directory
# ------------------------------------------------------------
mkdir -p "$EXP_DIR"
mkdir -p "$MODELS_SAVED"
mkdir -p "$LOGS_DIR"

log "Experiment: $EXP_NAME"
log "Results will be saved to: $EXP_DIR"

# Copy configs into experiment dir for reproducibility
cp "$CONFIG_MODEL" "$EXP_DIR/model_config.yaml"
cp "$CONFIG_TRAIN" "$EXP_DIR/training_config.yaml"

# ------------------------------------------------------------
# Stage 1 — Train the error detection model (CodeBERT)
# ------------------------------------------------------------
log "Stage 1/2 — Training error detection model (CodeBERT classifier)..."

RESUME_FLAG=""
if $RESUME; then
    RESUME_FLAG="--resume $MODELS_SAVED/codebert_model"
    log "Resuming from checkpoint: $MODELS_SAVED/codebert_model"
fi

python "$PROJECT_ROOT/src/training/train.py" \
    --model-config "$CONFIG_MODEL" \
    --training-config "$CONFIG_TRAIN" \
    --train-data "$DATA_PROCESSED/train.json" \
    --val-data "$DATA_PROCESSED/val.json" \
    --model-type error_detector \
    --output-dir "$MODELS_SAVED/codebert_model" \
    --exp-dir "$EXP_DIR" \
    --device "$DEVICE" \
    $RESUME_FLAG \
    2>&1 | tee "$LOGS_DIR/${EXP_NAME}_stage1.log"

success "Stage 1 complete. Checkpoint saved → $MODELS_SAVED/codebert_model"

# ------------------------------------------------------------
# Stage 2 — Fine-tune the explanation model (CodeT5 / LLaMA)
# ------------------------------------------------------------
log "Stage 2/2 — Fine-tuning explanation generation model..."

python "$PROJECT_ROOT/src/training/train.py" \
    --model-config "$CONFIG_MODEL" \
    --training-config "$CONFIG_TRAIN" \
    --train-data "$DATA_PROCESSED/train.json" \
    --val-data "$DATA_PROCESSED/val.json" \
    --model-type explanation \
    --output-dir "$MODELS_SAVED/explanation_model" \
    --exp-dir "$EXP_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$LOGS_DIR/${EXP_NAME}_stage2.log"

success "Stage 2 complete. Checkpoint saved → $MODELS_SAVED/explanation_model"

# ------------------------------------------------------------
# Save best combined checkpoint
# ------------------------------------------------------------
log "Linking best combined checkpoint..."

python "$PROJECT_ROOT/src/training/trainer.py" \
    --finalize \
    --error-detector-dir "$MODELS_SAVED/codebert_model" \
    --explanation-dir "$MODELS_SAVED/explanation_model" \
    --output "$MODELS_SAVED/best_mentor_model.pt" \
    2>&1 | tee -a "$LOGS_DIR/${EXP_NAME}_stage2.log"

success "Best model saved → $MODELS_SAVED/best_mentor_model.pt"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Training complete.${NC}"
echo -e "${GREEN} Experiment:  $EXP_NAME${NC}"
echo -e "${GREEN} Best model:  $MODELS_SAVED/best_mentor_model.pt${NC}"
echo -e "${GREEN} Logs:        $LOGS_DIR/${EXP_NAME}_stage1.log${NC}"
echo -e "${GREEN}              $LOGS_DIR/${EXP_NAME}_stage2.log${NC}"
echo -e "${GREEN} Next step:   bash scripts/evaluate.sh${NC}"
echo -e "${GREEN}============================================================${NC}"
