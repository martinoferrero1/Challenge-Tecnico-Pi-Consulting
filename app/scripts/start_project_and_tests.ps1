param(
    [int]$Port = 8000,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Using Python: $Python"

if (-not $SkipTests) {
    Write-Host "Running tests..."
    & $Python -m pytest
}

Write-Host "Starting API on http://127.0.0.1:$Port"
& $Python -m uvicorn app.main:app --reload --host 127.0.0.1 --port $Port
