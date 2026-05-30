@echo off
REM ============================================================================
REM One-click setup and launch for Vulnerability Scoring (Windows)
REM ============================================================================
REM Usage:
REM   scripts\setup.bat          -- setup venv + install deps (default)
REM   scripts\setup.bat train    -- setup + train models
REM   scripts\setup.bat service  -- setup + start API
REM   scripts\setup.bat docker   -- build and start Docker containers
REM   scripts\setup.bat docker --detach  -- build and start Docker containers in background
REM ============================================================================

cd /d "%~dp0.."

SET ACTION=%1
SET DOCKER_FLAG=%2

echo =========================================================================
echo   Vulnerability Scoring - Setup and Launch
echo =========================================================================
echo.

REM --- Step 1: Create .env if missing ----------------------------------------
IF NOT EXIST ".env" (
    echo [1/5] Creating .env from template...
    copy configs\.env.example .env >nul
    echo    OK .env created
) ELSE (
    echo [1/5] .env already exists, skipping
)

REM --- Docker path (no venv/pip needed) ----------------------------------------
IF /I "%ACTION%"=="docker" (
    echo [2/2] Building and starting Docker containers...
    where docker >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Docker Desktop not found.
        exit /b 1
    )
    docker compose version >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: docker compose not available.
        exit /b 1
    )
    IF /I "%DOCKER_FLAG%"=="--detach" (
        docker compose up --build -d
    ) ELSE (
        docker compose up --build
    )
    exit /b 0
)

REM --- Step 2: Create virtual environment ------------------------------------
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [2/5] Creating virtual environment...
    python -m venv .venv
    echo    OK venv created
) ELSE (
    echo [2/5] Virtual environment already exists, skipping
)

REM --- Step 3: Install CPU-only PyTorch FIRST (чтобы transformers не тянул CUDA-версию) ---
echo [3/5] Ensuring CPU-only PyTorch...
call .venv\Scripts\pip install --quiet --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps torch
echo    OK torch (CPU) installed

REM --- Step 4: Install remaining dependencies ----------------------------------
echo [4/5] Installing Python dependencies...
call .venv\Scripts\pip install --quiet -r requirements.txt
echo    OK requirements installed

REM --- Step 5: Run requested action ------------------------------------------
IF /I "%ACTION%"=="train" (
    echo [5/5] Training models...
    set "CUDA_VISIBLE_DEVICES=-1"
    set "TF_ENABLE_ONEDNN_OPTS=0"
    call .venv\Scripts\python -m src.train
) ELSE IF /I "%ACTION%"=="service" (
    echo [5/5] Starting API service...
    set "CUDA_VISIBLE_DEVICES=-1"
    set "TF_ENABLE_ONEDNN_OPTS=0"
    call .venv\Scripts\python -m src.service
) ELSE IF "%ACTION%"=="" (
    echo.
    echo =========================================================================
    echo   Setup complete. Next steps:
    echo     make train     - train models
    echo     make service   - start API
    echo     make docker    - build and run containers
    echo =========================================================================
) ELSE (
    echo Usage: %0 [train^|service^|docker]
)
