param(
    [ValidateSet('Audit', 'History', 'Sensitivity', 'All')]
    [string]$Mode = 'All',
    [string]$DataRoot = '',
    [string]$PriorResult = '',
    [string]$OutputTag = '',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
if (-not $Python) { $Python = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not $DataRoot) { $DataRoot = Join-Path $projectRoot 'data\friction_diverse_v1' }
if (-not $PriorResult) { $PriorResult = Join-Path $projectRoot 'result\friction_model_validation_full_v1' }
if (-not $OutputTag) { $OutputTag = Get-Date -Format 'yyyyMMdd_HHmmss' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot 'manifest.json') -PathType Leaf)) { throw "Dataset missing: $DataRoot" }
if (-not (Test-Path -LiteralPath (Join-Path $PriorResult 'overall.csv') -PathType Leaf)) { throw "Prior benchmark missing: $PriorResult" }

$jobs = @()
if ($Mode -in @('Audit', 'All')) {
    $jobs += @{ Module = 'src.model.friction_diagnostics'; Name = 'friction_model_diagnostics'; NeedsPrior = $true }
}
if ($Mode -in @('History', 'All')) {
    $jobs += @{ Module = 'src.model.friction_history_diagnostic'; Name = 'friction_history_diagnostic'; NeedsPrior = $false }
}
if ($Mode -in @('Sensitivity', 'All')) {
    $jobs += @{ Module = 'src.model.friction_kan_sensitivity'; Name = 'friction_kan_sensitivity'; NeedsPrior = $true }
}

Push-Location $projectRoot
try {
    foreach ($job in $jobs) {
        $output = Join-Path $projectRoot ('result\' + $job.Name + '_' + $OutputTag)
        $arguments = @('-m', $job.Module, '--data-root', $DataRoot, '--output', $output)
        if ($job.NeedsPrior) {
            $flag = if ($job.Name -eq 'friction_kan_sensitivity') { '--prior' } else { '--result' }
            $arguments += @($flag, $PriorResult)
        }
        Write-Host "Running $($job.Module) -> $output"
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) { throw "$($job.Module) failed with exit code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
