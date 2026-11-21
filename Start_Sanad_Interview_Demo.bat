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
if "%HF_HOME%"=="" if exist "A:\AI-Models\huggingface" set "HF_HOME=A:\AI-Models\huggingface"

set "QLORA_DIR=%~dp0..\qwen2.5-sanad-arabic-qlora"
set "QLORA_PY=%QLORA_DIR%\.venv\Scripts\python.exe"
set "QLORA_SCRIPT=%QLORA_DIR%\scripts\serve_local.py"

set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
set "SANAD_BACKEND_URL=http://127.0.0.1:8001"
set "SANAD_EMBEDDER=ollama"
set "SANAD_EMBED_MODEL=bge-m3"
set "SANAD_VECTOR_BACKEND=qdrant"
set "SANAD_TRACING=jsonl"
set "SANAD_CORS_ORIGINS=*"

echo [1/5] Checking Ollama server (for bge-m3 embeddings) on http://127.0.0.1:11434 ...
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=1.5)" >nul 2>&1
if errorlevel 1 (
    echo Starting headless Ollama server...
    start "Ollama Server" /min ollama serve
    timeout /t 4 /nobreak >nul
)

echo [2/5] Checking Mock Backend on http://127.0.0.1:8001 ...
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=1.5)" >nul 2>&1
if errorlevel 1 (
    echo Starting mock_backend on port 8001...
    start "Sanad Mock Backend" /min "%VENV_PY%" -m mock_backend
    timeout /t 2 /nobreak >nul
)

if not exist "%QLORA_PY%" goto :use_ollama
if not exist "%QLORA_SCRIPT%" goto :use_ollama

echo [3/5] Switching Sanad v3 LLM to fine-tuned Sanad QLoRA server (:8002/v1)...
ollama stop qwen2.5:7b >nul 2>&1
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=1.5)" >nul 2>&1
if errorlevel 1 (
    echo Launching Sanad QLoRA GPU server on port 8002...
    start "Sanad QLoRA LLM (:8002)" /min cmd /c "cd /d ""%QLORA_DIR%"" && ""%QLORA_PY%"" ""%QLORA_SCRIPT%"" --host 127.0.0.1 --port 8002"
    echo Waiting for Sanad QLoRA weights to load into GPU VRAM...
    for /L %%i in (1,1,150) do (
        "%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=1.5)" >nul 2>&1 && goto :qlora_ready
        timeout /t 2 /nobreak >nul
    )
)
:qlora_ready
set "SANAD_LLM=vllm"
set "SANAD_VLLM_BASE_URL=http://127.0.0.1:8002/v1"
set "SANAD_VLLM_MODEL=sanad-qwen25-ft"
goto :after_llm

:use_ollama
echo [3/5] Fine-tuned QLoRA venv not found; using Ollama qwen2.5:7b.
set "SANAD_LLM=ollama"
set "SANAD_CHAT_MODEL=qwen2.5:7b"

:after_llm

if not exist "data\index\qdrant" (
    echo [4/5] Indexing knowledge base into Qdrant using Ollama bge-m3...
    "%VENV_SANAD%" ingest
    if errorlevel 1 (
        echo [ERROR] sanad ingest failed.
        pause
        exit /b 1
    )
) else (
    echo [4/5] Qdrant index found at data\index\qdrant.
)

echo [5/5] Starting Sanad v3 server on http://127.0.0.1:8000 ...
echo Keep this window open during your interview.
echo - Full RAG UI:       http://127.0.0.1:8000
echo - Raw QLoRA UI:      http://127.0.0.1:8002
echo - Portfolio Live:    https://mohamed-eissa.pages.dev/projects/sanad
start "Open Sanad Browser" /min cmd /c "for /L %%i in (1,1,30) do (timeout /t 1 /nobreak >nul & "%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1.5)" >nul 2>&1 && (start "" "http://127.0.0.1:8000" & exit /b 0))"
"%VENV_SANAD%" serve --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo [ERROR] Sanad server exited with an error.
    pause
)
