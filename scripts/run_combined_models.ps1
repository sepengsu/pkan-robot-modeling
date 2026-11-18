param(
    [ValidateSet('Validate', 'Prepare', 'Train', 'Evaluate', 'All')]
    [string]$Stage = 'All',
    [string]$SourceRepo = 'D:\Code\sdml-doosan-robot',
    [string]$Python = '',
    [string]$OutputRoot = '',
    [string]$Config = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
if (-not $Python) { $Python = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot 'result\combined_legacy_wide_v3'
    if ($Stage -eq 'All' -and (Test-Path -LiteralPath $OutputRoot)) {
        $OutputRoot = "$OutputRoot`_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
    }
}
if (-not $Config) { $Config = Join-Path $projectRoot 'src\model\combined_config.json' }
$legacy = Join-Path $projectRoot 'data\grid16_legacy_joint_dataset_m0_m4'
$wide = Join-Path $projectRoot 'data\wide16_v3_joint_dataset'

foreach ($required in @($Python, $Config, (Join-Path $legacy 'dataset.npz'),
                        (Join-Path $wide 'dataset.npz'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file missing: $required"
    }
}
if (-not (Test-Path -LiteralPath $SourceRepo -PathType Container)) {
    throw "Source repository missing: $SourceRepo"
}

$common = @(
    '--legacy', $legacy,
    '--wide', $wide,
    '--source-repo', $SourceRepo,
    '--prepared', (Join-Path $OutputRoot 'prepared'),
    '--models', (Join-Path $OutputRoot 'models'),
    '--evaluation', (Join-Path $OutputRoot 'evaluation'),
    '--config', $Config
)
Write-Host "Output root: $OutputRoot"

function Invoke-ModelStage([string]$Name) {
    Write-Host "Running $Name ..."
    & $Python -m src.model.combined $Name.ToLowerInvariant() @common
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Push-Location $projectRoot
try {
    if ($Stage -eq 'All') {
        foreach ($name in @('Validate', 'Prepare', 'Train', 'Evaluate')) { Invoke-ModelStage $name }
    } else {
        Invoke-ModelStage $Stage
    }
} finally {
    Pop-Location
}
