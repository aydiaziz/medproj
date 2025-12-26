Param(
    [string]$Url
)

if (-not $Url -or $Url -eq "") {
    Write-Host "Usage: .\download_model.ps1 -Url <model_url>"
    Write-Host "Example: .\download_model.ps1 -Url https://storage.googleapis.com/your-bucket/hand_landmarker.task"
    exit 1
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelsDir = Join-Path $scriptRoot "models"
if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

$dest = Join-Path $modelsDir "hand_landmarker.task"
Write-Host "Downloading model from: $Url"

try {
    Invoke-WebRequest -Uri $Url -OutFile $dest -UseBasicParsing -ErrorAction Stop
    Write-Host "Model saved to: $dest"
} catch {
    Write-Error "Download failed: $_"
    exit 2
}
