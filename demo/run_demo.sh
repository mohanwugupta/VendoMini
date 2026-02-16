#!/bin/bash
# demo/run_demo.sh - Minimal reproduction script

# Ensure we are in the project root
cd "$(dirname "$0")/.." || exit

echo "--------------------------------------------------------"
echo "🛒  VendoMini — Prediction Error Crash Simulation Demo  "
echo "--------------------------------------------------------"

# 1. Setup virtualenv (optional for demo, good for isolation)
if [ ! -d "venv_demo" ]; then
    echo "[*] Creating virtual environment (venv_demo)..."
    python3 -m venv venv_demo
fi

source venv_demo/bin/activate

# 2. Install minimal dependencies
echo "[*] Installing minimal dependencies..."
pip install -r requirements.txt -q

# 3. Run the experiment
echo "[*] Running minimal simulation (10 steps)..."
# Using the mock model config we created
# Force output_dir to be results/demo in the command line to be safe, though config has it
python run_experiment.py --config demo/demo_config.yaml --n-jobs 1

# 4. Show results
echo ""
echo "--------------------------------------------------------"
echo "✅  Experiment Complete!"
echo "--------------------------------------------------------"

# The output might be in results/ or results/demo depending on how config is parsed
# Let's check both
TARGET_DIR="results/demo"
if [ ! -d "$TARGET_DIR" ]; then
    # Fallback if it wrote to root results
    if [ -f "results/aggregated_results.json" ]; then
        TARGET_DIR="results"
    fi
fi

ls -lh "$TARGET_DIR"
echo ""
echo "[*] Sample of the summary.json output:"

# Check for aggregated results first
if [ -f "$TARGET_DIR/aggregated_results.json" ]; then
    echo "Found aggregated results in $TARGET_DIR/aggregated_results.json"
    cat "$TARGET_DIR/aggregated_results.json"
else
    # Find the latest run folder
    LATEST_RUN=$(ls -dt $TARGET_DIR/run_*/ | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "Found run in: $LATEST_RUN"
        cat "${LATEST_RUN}summary.json"
    else
        echo "No results found to display."
    fi
fi

echo ""
echo "[*] Demo finished successfully."
