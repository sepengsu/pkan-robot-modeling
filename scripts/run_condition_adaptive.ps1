param(
    [ValidateSet('All', 'Audit', 'Basis', 'Train', 'Figures')]
    [string]$Stage = 'All',
    [string]$OutputRoot = 'F:\robot_modeling\result\condition_adaptive_rbd_kan_v1',
    [string]$Python = 'F:\robot_modeling\.venv\Scripts\python.exe',
    [string]$SourceRepo = 'D:\Code\sdml-doosan-robot'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath $SourceRepo -PathType Container)) { throw "Source repository missing: $SourceRepo" }
if ($Stage -eq 'All' -and (Test-Path -LiteralPath $OutputRoot)) {
    throw "Result folder already exists: $OutputRoot"
}
Push-Location $projectRoot
try {
    & $Python -m src.model.condition_adaptive $Stage.ToLowerInvariant() --output $OutputRoot --source-repo $SourceRepo
    if ($LASTEXITCODE -ne 0) { throw "Study failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
