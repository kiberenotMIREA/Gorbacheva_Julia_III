#!/usr/bin/env bash
# =============================================================================
# Cleanup script for Vulnerability Scoring project
# Removes all generated artifacts: models, processed data, cache, node_modules
# After cleanup, retrain with: python -m src.train
# =============================================================================

set -euo pipefail

echo "========================================================================="
echo "  Vulnerability Scoring - Cleanup"
echo "========================================================================="
echo ""

# ------------------------------------------------------------------
# 1. Remove saved models
# ------------------------------------------------------------------
echo "[1/10] Removing models..."
if [ -d "artifacts/models" ]; then
    rm -f artifacts/models/*.keras 2>/dev/null
    rm -f artifacts/models/*.h5 2>/dev/null
    rm -f artifacts/models/*.pkl 2>/dev/null
    rm -f artifacts/models/*.json 2>/dev/null
    find artifacts/models -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
    echo "   OK Models removed"
else
    echo "   - Directory artifacts/models not found"
fi

# ------------------------------------------------------------------
# 2. Remove processed data
# ------------------------------------------------------------------
echo "[2/10] Removing processed data..."
if [ -d "data/processed" ]; then
    rm -f data/processed/*.pkl 2>/dev/null
    rm -f data/processed/*.csv 2>/dev/null
    echo "   OK Processed data removed"
else
    echo "   - Directory data/processed not found"
fi

# ------------------------------------------------------------------
# 3. Clean Python cache (__pycache__)
# ------------------------------------------------------------------
echo "[3/10] Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "   OK Python cache cleaned"

# ------------------------------------------------------------------
# 4. Remove MLflow tracking logs (artifacts/mlruns)
# ------------------------------------------------------------------
echo "[4/10] Removing MLflow tracking logs..."
if [ -d "artifacts/mlruns" ]; then
    rm -rf artifacts/mlruns
    echo "   OK MLflow tracking logs removed"
else
    echo "   - MLflow tracking logs not found"
fi

# ------------------------------------------------------------------
# 5. Clean test cache
# ------------------------------------------------------------------
echo "[5/10] Cleaning test cache..."
if [ -d ".pytest_cache" ]; then
    rm -rf .pytest_cache
    echo "   OK Test cache cleaned"
else
    echo "   - Test cache not found"
fi

# ------------------------------------------------------------------
# 6. Remove training logs
# ------------------------------------------------------------------
echo "[6/10] Removing training logs..."
if [ -d "artifacts/logs" ]; then
    rm -rf artifacts/logs
    echo "   OK Training logs removed"
fi

# ------------------------------------------------------------------
# 7. Remove figures
# ------------------------------------------------------------------
echo "[7/10] Removing figures..."
if [ -d "artifacts/figures" ]; then
    rm -f artifacts/figures/* 2>/dev/null
    echo "   OK Figures removed"
fi

# ------------------------------------------------------------------
# 8. Remove fine-tuned BERT
# ------------------------------------------------------------------
echo "[8/10] Removing fine-tuned BERT..."
if [ -d "artifacts/models/bert-finetuned" ]; then
    rm -rf artifacts/models/bert-finetuned
    echo "   OK Fine-tuned BERT removed"
fi

# ------------------------------------------------------------------
# 9. Remove frontend node_modules
# ------------------------------------------------------------------
echo "[9/10] Removing frontend node_modules..."
if [ -d "frontend/node_modules" ]; then
    echo "   Removing frontend/node_modules..."
    rm -rf frontend/node_modules
    echo "   OK frontend node_modules removed"
else
    echo "   - Directory frontend/node_modules not found"
fi

# ------------------------------------------------------------------
# 10. Remove cached raw dataset (data/raw/vuldeepecker_raw.pkl)
# ------------------------------------------------------------------
echo "[10/10] Removing cached raw dataset..."
if [ -f "data/raw/vuldeepecker_raw.pkl" ]; then
    rm -f data/raw/vuldeepecker_raw.pkl
    echo "   OK Cached dataset removed"
else
    echo "   - Cached dataset not found"
fi

echo ""
echo "========================================================================="
echo "  Cleanup complete."
echo ""
echo "  To retrain, run: python -m src.train"
echo "========================================================================="
