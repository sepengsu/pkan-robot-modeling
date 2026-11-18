param(
    [switch]$Preview,
    [switch]$Smoke,
    [string]$OutputRoot = '',
    [string]$SourceRepo = 'D:\Code\sdml-doosan-robot',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
if (-not $Python) { $Python = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not $OutputRoot) {
    if ($Smoke) {
        $OutputRoot = Join-Path $projectRoot ('result\friction_diverse_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    } else {
        $OutputRoot = Join-Path $projectRoot 'data\friction_diverse_v1'
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath $SourceRepo -PathType Container)) { throw "Source repo missing: $SourceRepo" }

$command = if ($Preview) { 'plan' } else { 'collect' }
$arguments = @('-m', 'src.model.collect_friction', $command,
               '--source-repo', $SourceRepo, '--output', $OutputRoot)
if ($Smoke) {
    $arguments += @('--condition', 'ag145_p0', '--trajectory', 'T1',
                    '--max-rows-per-trial', '100')
}

Write-Host "Output: $OutputRoot"
Write-Host "Trials: 3 distinct seeded trajectory perturbations"
Push-Location $projectRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Friction collection failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
