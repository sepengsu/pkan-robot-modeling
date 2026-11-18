param(
    [switch]$Smoke,
    [string]$DataRoot = '',
    [string]$OutputRoot = '',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
if (-not $Python) { $Python = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not $DataRoot) { $DataRoot = Join-Path $projectRoot 'data\friction_diverse_v1' }
if (-not $OutputRoot) {
    $name = if ($Smoke) { 'friction_model_validation_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss') } else { 'friction_model_validation_full_v1' }
    $OutputRoot = Join-Path $projectRoot ('result\' + $name)
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot 'manifest.json') -PathType Leaf)) { throw "Dataset manifest missing: $DataRoot" }
$arguments = @('-m', 'src.model.friction_validate', '--data-root', $DataRoot, '--output', $OutputRoot)
if ($Smoke) { $arguments += '--smoke' }
Push-Location $projectRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Validation failed with exit code $LASTEXITCODE" }
    & $Python -m src.model.plot_friction_validation --result $OutputRoot
    if ($LASTEXITCODE -ne 0) { throw "Validation plots failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
