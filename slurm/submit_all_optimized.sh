#!/bin/bash

# Master submission script for VendoMini experiments
# Submits small and large model jobs separately for optimal resource allocation

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  VendoMini Experiment Submission (Optimized Resource Usage)   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Function to submit a phase
submit_phase() {
    local phase=$1
    local model_size=$2
    local script="slurm/run_${phase}_${model_size}.sh"
    
    if [ -f "$script" ]; then
        echo "📤 Submitting ${phase} (${model_size} models)..."
        job_id=$(sbatch "$script" | awk '{print $4}')
        echo "   ✅ Job ID: $job_id"
        return 0
    else
        echo "   ⚠️  Script not found: $script"
        return 1
    fi
}

# Parse command line arguments
MODE="${1:-all}"  # all, small, large, or specific phase

echo "Mode: $MODE"
echo ""

case $MODE in
    small)
        echo "🔹 Submitting SMALL model jobs only (1 GPU, fast queue)..."
        echo "   Models: deepseek-llm-7b (7B), gpt-oss-20b (20B)"
        echo ""
        submit_phase "phase1" "small"
        submit_phase "phase2" "small"
        submit_phase "phase3" "small"
        submit_phase "phase4" "small"
        submit_phase "phase5" "small"
        ;;
    
    large)
        echo "🔸 Submitting LARGE model jobs only (2 GPUs, slower queue)..."
        echo "   Models: Qwen3-32B, Llama-3.3-70B, Qwen2.5-72B, DeepSeek-V2.5"
        echo ""
        submit_phase "phase1" "large"
        submit_phase "phase2" "large"
        submit_phase "phase3" "large"
        submit_phase "phase4" "large"
        submit_phase "phase5" "large"
        ;;
    
    all)
        echo "🚀 Submitting ALL jobs (small first for faster results)..."
        echo ""
        echo "═══ SMALL MODELS (1 GPU) ═══"
        submit_phase "phase1" "small"
        submit_phase "phase2" "small"
        submit_phase "phase3" "small"
        submit_phase "phase4" "small"
        submit_phase "phase5" "small"
        echo ""
        echo "═══ LARGE MODELS (2 GPUs) ═══"
        submit_phase "phase1" "large"
        submit_phase "phase2" "large"
        submit_phase "phase3" "large"
        submit_phase "phase4" "large"
        submit_phase "phase5" "large"
        ;;
    
    phase*)
        # Submit specific phase (both small and large)
        phase_num=${MODE#phase}
        echo "📋 Submitting Phase ${phase_num} (both small and large models)..."
        echo ""
        submit_phase "phase${phase_num}" "small"
        submit_phase "phase${phase_num}" "large"
        ;;
    
    test)
        echo "🧪 TEST MODE: Submitting Phase 1 small models only..."
        echo "   Use this to verify setup before full run"
        echo ""
        submit_phase "phase1" "small"
        ;;
    
    *)
        echo "❌ Invalid mode: $MODE"
        echo ""
        echo "Usage: ./submit_all_optimized.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all     - Submit all phases (small + large models)"
        echo "  small   - Submit only small model jobs (7B, 20B) - FAST QUEUE"
        echo "  large   - Submit only large model jobs (32B-236B) - SLOW QUEUE"
        echo "  test    - Submit phase1 small models only (for testing)"
        echo "  phase1  - Submit specific phase (both sizes)"
        echo "  phase2  - Submit specific phase (both sizes)"
        echo "  ...etc"
        echo ""
        echo "Resource Allocation:"
        echo "  Small models: 1 GPU, 48GB RAM, 2 CPUs, ~2 hours"
        echo "  Large models: 2 GPUs, 128GB RAM, 4 CPUs, ~6 hours"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Submission complete!"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo "═══════════════════════════════════════════════════════════════"
