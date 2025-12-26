# run_backend.ps1
# Lancer le backend depuis la racine du dépôt pour éviter "ModuleNotFoundError: No module named 'backend'"
# Utilisation (PowerShell) :
#   .\run_backend.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# Si vous utilisez un venv, activez-le d'abord si nécessaire, p.ex. :
# & .\.venv\Scripts\Activate.ps1

# Lancer uvicorn depuis la racine en important le module "backend.app:app"
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
