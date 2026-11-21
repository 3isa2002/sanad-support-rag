@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "VENV_SANAD=%CD%\.venv\Scripts\sanad.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] .venv not found. Create it with: python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev]"
    pause
    exit /b 1
)

if "%OLLAMA_MODELS%"=="" if exist "A:\AI-Models\ollama" set "OLLAMA_MODELS=A:\AI-Models\ollama"

set "SANAD_EMBEDDER=ollama"
set "SANAD_EMBED_MODEL=bge-m3"
set "SANAD_LLM=ollama"
set "SANAD_CHAT_MODEL=qwen2.5:7b"
set "SANAD_VECTOR_BACKEND=qdrant"
set "SANAD_TRACING=jsonl"
set "SANAD_CORS_ORIGINS=*"

echo [1/4] Checking Ollama server on http://127.0.0.1:11434 ...
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=1.5)" >nul 2>&1
if errorlevel 1 (
    echo Starting headless Ollama server...
    start "Ollama Server" /min ollama serve
    timeout /t 4 /nobreak >nul
)

echo [2/4] Checking Mock Backend on http://127.0.0.1:8001 ...
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=1.5)" >nul 2>&1
if errorlevel 1 (
    echo Starting mock_backend on port 8001...
    start "Sanad Mock Backend" /min "%VENV_PY%" -m mock_backend
    timeout /t 2 /nobreak >nul
)

if not exist "data\index\qdrant" (
    echo [3/4] Indexing knowledge base into Qdrant using Ollama bge-m3...
    "%VENV_SANAD%" ingest
    if errorlevel 1 (
        echo [ERROR] sanad ingest failed. Check Ollama status above.
        pause
        exit /b 1
    )
) else (
    echo [3/4] Qdrant index found at data\index\qdrant.
)

echo [4/4] Starting Sanad v3 server on http://127.0.0.1:8000 ...
echo Keep this window open during your interview.
echo Connect https://mohamed-eissa.pages.dev/projects/sanad (Live Backend) OR open http://127.0.0.1:8000
start "Open Sanad Browser" /min cmd /c "for /L %%i in (1,1,25) do (timeout /t 1 /nobreak >nul & "%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)" >nul 2>&1 && (start "" "http://127.0.0.1:8000" & exit /b 0))"
"%VENV_SANAD%" serve --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo [ERROR] Sanad server exited with an error.
    pause
)
