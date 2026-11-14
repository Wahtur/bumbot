#!/usr/bin/env bash
# run_experiments.sh
# Automated Training & Evaluation Suite for TD-Leaf Classical Chess Bot
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PYTHON="$DIR/venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

GAMES=${1:-200}
WORKERS=${2:-8}
DEPTH=${3:-2}

echo "=========================================================="
echo " Starting TD-Leaf Experiments on wahtur-homeserver"
echo " Games: $GAMES | Workers: $WORKERS | Search Depth: $DEPTH"
echo "=========================================================="

mkdir -p checkpoints plots logs

# 1. Run Self-Play Training
echo ""
echo "[Step 1/3] Launching TD-Leaf Self-Play Training..."
$PYTHON td_leaf_trainer.py \
    --games "$GAMES" \
    --workers "$WORKERS" \
    --depth "$DEPTH" \
    --lr 0.001 \
    --lambda-val 0.7 \
    --checkpoint-interval 50 \
    --save-dir checkpoints \
    --log-file training_log.csv

# 2. Run Tournament Evaluation
echo ""
echo "[Step 2/3] Evaluating Trained Model against Baselines..."
LATEST_CKPT=$(ls -t checkpoints/weights_game_*.pt | head -n 1)

echo "Evaluating $LATEST_CKPT vs RandomBot (20 games)..."
$PYTHON evaluate.py --checkpoint "$LATEST_CKPT" --opponent random --games 20 --depth 2

echo "Evaluating $LATEST_CKPT vs GreedyBot (20 games)..."
$PYTHON evaluate.py --checkpoint "$LATEST_CKPT" --opponent greedy --games 20 --depth 2

# 3. Generate Visualizations
echo ""
echo "[Step 3/3] Generating Telemetry & Weight Plots..."
$PYTHON plot_results.py --log training_log.csv --checkpoint "$LATEST_CKPT" --out-dir plots

echo ""
echo "=========================================================="
echo " Experiment Completed Successfully!"
echo " Results available in training_log.csv and plots/"
echo "=========================================================="
