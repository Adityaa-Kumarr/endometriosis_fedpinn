# Run the Streamlit app using the venv on H: (no C: drive usage for packages).
# Usage: .\run_app.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found. Create it first (all on H:):"
    Write-Host "  py -m venv $ProjectRoot\venv"
    Write-Host "  & $ProjectRoot\venv\Scripts\Activate.ps1; pip install -r $ProjectRoot\requirements.txt"
    exit 1
}

Set-Location $ProjectRoot
& $VenvPython -m streamlit run app.py @args
