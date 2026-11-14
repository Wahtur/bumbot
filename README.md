# Bumbot: Classical Reinforcement Learning Chess Bot (TD-Leaf)

A high-performance classical machine learning chess engine utilizing **TD-Leaf($\lambda$)** reinforcement learning (Baxter et al., 1998) and adversarial Minimax search with Alpha-Beta pruning.

---

## Architecture & Project Structure

The project is architected as two distinct sub-projects:

- **Mini-Project 1: Search & Representation (Partner A)**
  - `feature_extractor.py`: Extracts 651 handcrafted features (Material imbalances, 640-dim Piece-Square Tables, Mobility, Center Control, Pawn Structure, King Safety) using 64-bit bitboard operations.
  - `search_engine.py`: Minimax with Alpha-Beta pruning, MVV-LVA move ordering, and Principal Variation leaf tracking.

- **Mini-Project 2: Machine Learning & Self-Play (Partner B)**
  - `td_leaf_trainer.py`: Baxter et al.'s TD-Leaf($\lambda$) implementation using PyTorch autograd, hyperbolic tangent squashing, eligibility traces, and multi-core parallel rollout generation.
  - `evaluate.py`: Automated benchmarking tournament vs RandomBot, GreedyBot, and HeuristicBot with empirical Elo rating estimation.
  - `plot_results.py`: Telemetry visualization (loss curves, game durations, parameter norm, and learned feature weight bar charts).

---

## Quick Start (Workstation)

### 1. Run a Quick Test Game
```bash
python evaluate.py --opponent random --games 2 --depth 1
```

### 2. Run Local Training
```bash
python td_leaf_trainer.py --games 10 --workers 2 --depth 1
```

### 3. Generate Visualizations
```bash
python plot_results.py
```

---

## Homeserver Training (`wahtur-homeserver`)

The homeserver environment (`192.168.1.135`, 12 CPU cores, 32GB RAM) is pre-configured with Python 3.12 and dependencies.

### 1. SSH into the Homeserver
```bash
ssh homeserver
cd /home/antigravity/bumbot
```

### 2. Run Experiments (Headless / Multiprocessing)
```bash
# Runs 200 self-play games across 8 CPU cores at search depth 2,
# evaluates against baselines, and generates plots
./run_experiments.sh 200 8 2
```

### 3. Fetch Telemetry and Plots back to Workstation
```powershell
scp -r homeserver:/home/antigravity/bumbot/plots ./
scp homeserver:/home/antigravity/bumbot/training_log.csv ./
```
