#!/usr/bin/env bash
# ============================================================
# PolyMentor — Interactive Tutor Script
# Usage: bash scripts/run_tutor.sh [--model PATH] [--level LEVEL] [--lang LANGUAGE] [--api]
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Colours
# ------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo -e "${CYAN}[TUTOR]${NC} $1"; }
success(){ echo -e "${GREEN}[READY]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
MODEL_PATH=""
LEVEL="beginner"
LANGUAGE="python"
API_MODE=false
API_HOST="0.0.0.0"
API_PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL_PATH="$2";  shift 2 ;;
        --level)    LEVEL="$2";       shift 2 ;;
        --lang)     LANGUAGE="$2";    shift 2 ;;
        --api)      API_MODE=true;    shift ;;
        --host)     API_HOST="$2";    shift 2 ;;
        --port)     API_PORT="$2";    shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_SAVED="$PROJECT_ROOT/models_saved"
LOGS_DIR="$PROJECT_ROOT/experiments/logs"

# Default model path
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODELS_SAVED/best_mentor_model.pt"
fi

# ------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------
log "Validating environment..."

python --version >/dev/null 2>&1 || error "Python not found."
python -c "import torch; import transformers" >/dev/null 2>&1 \
    || error "Dependencies missing. Run: pip install -r requirements.txt"
python -c "import src" >/dev/null 2>&1 \
    || error "Project not installed. Run: pip install -e ."

[[ -f "$MODEL_PATH" ]] \
    || error "Model not found at: $MODEL_PATH\nRun: bash scripts/train.sh first."

# Validate level
case "$LEVEL" in
    beginner|intermediate|advanced) ;;
    *) error "Invalid level: $LEVEL. Choose from: beginner, intermediate, advanced" ;;
esac

# Validate language
case "$LANGUAGE" in
    python|javascript|cpp|java) ;;
    *) error "Invalid language: $LANGUAGE. Choose from: python, javascript, cpp, java" ;;
esac

mkdir -p "$LOGS_DIR"

# ------------------------------------------------------------
# API Server Mode
# ------------------------------------------------------------
if $API_MODE; then
    log "Starting PolyMentor API server..."
    log "  Host:     $API_HOST"
    log "  Port:     $API_PORT"
    log "  Model:    $MODEL_PATH"
    log ""
    log "Endpoint: POST http://$API_HOST:$API_PORT/analyze"
    log "Docs:     http://$API_HOST:$API_PORT/docs"
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN} PolyMentor API is starting...${NC}"
    echo -e "${GREEN} Press Ctrl+C to stop.${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""

    POLYMENTOR_MODEL_PATH="$MODEL_PATH" \
    python -m uvicorn src.inference.pipeline:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        2>&1 | tee "$LOGS_DIR/api_server.log"

    exit 0
fi

# ------------------------------------------------------------
# Interactive CLI Tutor Mode
# ------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ██████╗  ██████╗ ██╗  ██╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗ ██████╗ ██████╗ "
echo "  ██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗██╔══██╗"
echo "  ██████╔╝██║   ██║██║   ╚████╔╝ ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║██████╔╝"
echo "  ██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██╔══██╗"
echo "  ██║     ╚██████╔╝███████╗██║   ██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ╚██████╔╝██║  ██║"
echo "  ╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "  ${CYAN}Don't just fix code. Understand it.${NC}"
echo ""
echo -e "  Model    : ${BOLD}$MODEL_PATH${NC}"
echo -e "  Language : ${BOLD}$LANGUAGE${NC}"
echo -e "  Level    : ${BOLD}$LEVEL${NC}"
echo ""
echo -e "  ${YELLOW}Commands:${NC}"
echo -e "    ${BOLD}:lang <python|javascript|cpp|java>${NC}  — switch language"
echo -e "    ${BOLD}:level <beginner|intermediate|advanced>${NC} — switch level"
echo -e "    ${BOLD}:hint${NC}                               — get the next hint"
echo -e "    ${BOLD}:reset${NC}                              — start a new session"
echo -e "    ${BOLD}:quit${NC}  or  Ctrl+C                   — exit"
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""

log "Loading model... (this may take a moment)"

python "$PROJECT_ROOT/src/inference/tutor_mode.py" \
    --model "$MODEL_PATH" \
    --language "$LANGUAGE" \
    --level "$LEVEL" \
    --log-file "$LOGS_DIR/tutor_session_$(date +%Y%m%d_%H%M%S).log" \
    2>&1

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}Session ended. Happy coding!${NC}"
else
    echo -e "${RED}Session ended with an error. Check $LOGS_DIR for details.${NC}"
fi
