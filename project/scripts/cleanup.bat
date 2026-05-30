@echo off
REM ===========================================================================
REM Cleanup script for Vulnerability Scoring project (Windows)
REM Removes all generated artifacts: models, processed data, cache, node_modules
REM After cleanup, retrain with: python -m src.train
REM ===========================================================================

echo =========================================================================
echo   Vulnerability Scoring - Cleanup
echo =========================================================================
echo.

REM --- Step 1: Remove trained models -----------------------------------------
echo [1/10] Removing models...
if exist "artifacts\models" (
    if exist "artifacts\models\*.keras" del /q "artifacts\models\*.keras" 2>nul
    if exist "artifacts\models\*.h5" del /q "artifacts\models\*.h5" 2>nul
    if exist "artifacts\models\*.pkl" del /q "artifacts\models\*.pkl" 2>nul
    if exist "artifacts\models\*.json" del /q "artifacts\models\*.json" 2>nul
    for /d %%d in (artifacts\models\*) do if exist "%%d" rmdir /s /q "%%d" 2>nul
    echo    OK Models removed
) else (
    echo    - Directory artifacts\models not found
)

REM --- Step 2: Remove processed data -----------------------------------------
echo [2/10] Removing processed data...
if exist "data\processed" (
    if exist "data\processed\*.pkl" del /q "data\processed\*.pkl" 2>nul
    if exist "data\processed\*.csv" del /q "data\processed\*.csv" 2>nul
    echo    OK Processed data removed
) else (
    echo    - Directory data\processed not found
)

REM --- Step 3: Clean Python cache --------------------------------------------
echo [3/10] Cleaning Python cache...
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
echo    OK Python cache cleaned

REM --- Step 4: Remove MLflow tracking logs -----------------------------------
echo [4/10] Removing MLflow tracking logs...
if exist "artifacts\mlruns" (
    rmdir /s /q "artifacts\mlruns" 2>nul
    echo    OK MLflow tracking logs removed
) else (
    echo    - MLflow tracking logs not found
)

REM --- Step 5: Clean test cache ----------------------------------------------
echo [5/10] Cleaning test cache...
if exist ".pytest_cache" rmdir /s /q ".pytest_cache" && echo    OK Test cache cleaned || echo    - Test cache not found

REM --- Step 6: Remove training logs ------------------------------------------
echo [6/10] Removing training logs...
if exist "artifacts\logs" rmdir /s /q "artifacts\logs" && echo    OK Training logs removed

REM --- Step 7: Remove figures ------------------------------------------------
echo [7/10] Removing figures...
if exist "artifacts\figures" (
    del /q "artifacts\figures\*" 2>nul
    echo    OK Figures removed
)

REM --- Step 8: Remove fine-tuned BERT ----------------------------------------
echo [8/10] Removing fine-tuned BERT...
if exist "artifacts\models\bert-finetuned" (
    rmdir /s /q "artifacts\models\bert-finetuned" 2>nul
    echo    OK Fine-tuned BERT removed
)

REM --- Step 9: Remove frontend node_modules ----------------------------------
echo [9/10] Removing frontend node_modules...
if exist "frontend\node_modules" (
    rmdir /s /q "frontend\node_modules" 2>nul
    echo    OK frontend node_modules removed
) else (
    echo    - Directory frontend\node_modules not found
)

REM --- Step 10: Remove cached raw dataset ------------------------------------
echo [10/10] Removing cached raw dataset...
if exist "data\raw\vuldeepecker_raw.pkl" (
    del /q "data\raw\vuldeepecker_raw.pkl" 2>nul
    echo    OK Cached dataset removed
) else (
    echo    - Cached dataset not found
)

echo.
echo =========================================================================
echo   Cleanup complete.
echo.
echo   To retrain, run: python -m src.train
echo =========================================================================
