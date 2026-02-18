#!/bin/bash
# demo/run_demo.sh — Customer order fulfillment demo

# Ensure we are in the project root
cd "$(dirname "$0")/.." || exit

echo "============================================================"
echo "🛒  VendoMini — Customer Order Fulfillment Demo             "
echo "============================================================"
echo ""
echo "Goal: Procure inventory from suppliers, ship to customers,"
echo "      earn revenue, and avoid budget ruin or too many"
echo "      expired customer orders."
echo ""

# 1. Setup virtualenv (optional for demo, good for isolation)
if [ ! -d "venv_demo" ]; then
    echo "[*] Creating virtual environment (venv_demo)..."
    python3 -m venv venv_demo
fi

source venv_demo/bin/activate

# 2. Install minimal dependencies
echo "[*] Installing dependencies..."
pip install -r requirements.txt -q

# 3. Run the experiment
echo ""
echo "[*] Running simulation (15 steps, mock LLM, no API key required)..."
python run_experiment.py --config demo/demo_config.yaml --n-jobs 1

echo ""
echo "============================================================"
echo "✅  Simulation Complete!"
echo "============================================================"

# 4. Find and display results
LOGS_DIR="logs"
LATEST_RUN=$(ls -dt "${LOGS_DIR}"/run_*/ 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    SUMMARY="${LATEST_RUN}summary.json"
    echo ""
    echo "📁 Run folder: ${LATEST_RUN}"
    echo ""

    if [ -f "$SUMMARY" ]; then
        echo "📊 Run Summary:"
        echo "------------------------------------------------------------"
        # Use python to pretty-print the relevant fields
        python3 - "$SUMMARY" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    s = json.load(f)

budget  = s.get('final_budget', 0)
revenue = s.get('revenue', 0)
shipped = s.get('customer_orders_shipped', 0)
failed  = s.get('customer_orders_failed', 0)
steps   = s.get('total_steps', 0)
crashed = s.get('crashed', False)
ctype   = s.get('crash_type', 'N/A')

print(f"  Steps completed          : {steps}")
print(f"  Final budget             : ${budget:.2f}")
print(f"  Total revenue earned     : ${revenue:.2f}")
print(f"  Customer orders shipped  : {shipped}")
print(f"  Customer orders failed   : {failed}  (expired without being shipped)")
print(f"  Simulation ended (crash) : {crashed}  [{ctype}]")
PYEOF
        echo "------------------------------------------------------------"
        echo ""
        echo "💡 Key insight:"
        echo "   Revenue is earned each time the agent ships a customer order."
        echo "   The agent loses if budget < -\$100 OR ≥10 orders expire."
    else
        echo "(summary.json not found in ${LATEST_RUN})"
    fi
else
    echo "(No run folders found in ${LOGS_DIR}/)"
fi

echo ""
echo "[*] Demo finished."

