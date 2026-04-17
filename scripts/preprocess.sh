#!/usr/bin/env bash
# ============================================================
# PolyMentor — Data Preprocessing Script
# Usage: bash scripts/preprocess.sh
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Colours for readable output
# ------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

log()    { echo -e "${CYAN}[PREPROCESS]${NC} $1"; }
success(){ echo -e "${GREEN}[DONE]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW="$PROJECT_ROOT/data/raw"
DATA_PROCESSED="$PROJECT_ROOT/data/processed"
DATA_LABELS="$PROJECT_ROOT/data/labels"
LOGS_DIR="$PROJECT_ROOT/experiments/logs"

# ------------------------------------------------------------
# Validate environment
# ------------------------------------------------------------
log "Checking environment..."

python --version >/dev/null 2>&1 || error "Python not found. Install Python 3.10+."
python -c "import src" >/dev/null 2>&1 || error "Project not installed. Run: pip install -e ."

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python version: $PYTHON_VERSION"

# ------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------
log "Creating directory structure..."

mkdir -p "$DATA_RAW/code_datasets"
mkdir -p "$DATA_RAW/error_samples"
mkdir -p "$DATA_RAW/programming_questions"
mkdir -p "$DATA_PROCESSED"
mkdir -p "$DATA_LABELS"
mkdir -p "$LOGS_DIR"

success "Directories ready."

# ------------------------------------------------------------
# Step 1 — Collect raw data
# ------------------------------------------------------------
log "Step 1/4 — Collecting raw data..."

python "$PROJECT_ROOT/src/data_pipeline/collector.py" \
    --output-dir "$DATA_RAW" \
    --languages python javascript cpp java \
    --sources codenet stackoverflow leetcode manybugs custom \
    2>&1 | tee "$LOGS_DIR/collect.log"

success "Raw data collected → $DATA_RAW"

# ------------------------------------------------------------
# Step 2 — Clean and normalise
# ------------------------------------------------------------
log "Step 2/4 — Cleaning and normalising data..."

python "$PROJECT_ROOT/src/data_pipeline/cleaner.py" \
    --input-dir "$DATA_RAW" \
    --output-dir "$DATA_RAW/cleaned" \
    --min-lines 3 \
    --max-tokens 512 \
    2>&1 | tee "$LOGS_DIR/clean.log"

success "Data cleaned → $DATA_RAW/cleaned"

# ------------------------------------------------------------
# Step 3 — Tokenise
# ------------------------------------------------------------
log "Step 3/4 — Tokenising code..."

python "$PROJECT_ROOT/src/data_pipeline/tokenizer.py" \
    --input-dir "$DATA_RAW/cleaned" \
    --output-dir "$DATA_RAW/tokenized" \
    --config "$PROJECT_ROOT/configs/language_config.yaml" \
    2>&1 | tee "$LOGS_DIR/tokenize.log"

success "Tokenisation complete → $DATA_RAW/tokenized"

# ------------------------------------------------------------
# Step 4 — Build train / val / test splits
# ------------------------------------------------------------
log "Step 4/4 — Building dataset splits..."

python "$PROJECT_ROOT/src/data_pipeline/dataset_builder.py" \
    --input-dir "$DATA_RAW/tokenized" \
    --output-dir "$DATA_PROCESSED" \
    --labels-dir "$DATA_LABELS" \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    2>&1 | tee "$LOGS_DIR/build.log"

success "Dataset splits written:"
log "  → $DATA_PROCESSED/train.json"
log "  → $DATA_PROCESSED/val.json"
log "  → $DATA_PROCESSED/test.json"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Preprocessing complete.${NC}"
echo -e "${GREEN} Logs saved to: $LOGS_DIR${NC}"
echo -e "${GREEN} Next step: bash scripts/train.sh${NC}"
echo -e "${GREEN}============================================================${NC}"
