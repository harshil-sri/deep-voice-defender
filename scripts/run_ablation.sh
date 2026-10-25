#!/bin/bash
# Ablation study runner: 4 modes x 3 seeds = 12 trainings
# Usage: bash scripts/run_ablation.sh

set -e

MODES=("none" "opus" "noise" "full")
SEEDS=(42 123 456)

echo "=========================================="
echo "DEEP-VOICE DEFENDER ABLATION STUDY"
echo "Modes: ${MODES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#MODES[@]} * ${#SEEDS[@]}))"
echo "=========================================="

cd "$(dirname "$0")/.."
source venv/bin/activate

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        output="deep_voice_defender_${mode}_seed${seed}.keras"
        
        if [ -f "$output" ]; then
            echo "⏭️  Skipping $output (already exists)"
            continue
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🏃 Training: mode=$mode, seed=$seed"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python src/train.py --aug_mode "$mode" --seed "$seed" --output "$output"
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $output"
        else
            echo "❌ Failed: $output"
            exit 1
        fi
    done
done

echo ""
echo "=========================================="
echo "ALL TRAININGS COMPLETE"
echo "=========================================="
ls -la deep_voice_defender_*_seed*.keras