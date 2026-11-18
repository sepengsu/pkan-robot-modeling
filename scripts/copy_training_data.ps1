param(
    [string]$SourceRoot = 'D:\Code\sdml-doosan-robot',
    [string]$DestinationData = ''
)

$ErrorActionPreference = 'Stop'

$sourceRootPath = (Resolve-Path -LiteralPath $SourceRoot).ProviderPath
if (-not $DestinationData) {
    $DestinationData = Join-Path (Split-Path -Parent $PSScriptRoot) 'data'
}
$destinationRootPath = [System.IO.Path]::GetFullPath($DestinationData)
if ($destinationRootPath.StartsWith($sourceRootPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'DestinationData must be outside SourceRoot.'
}

$plan = [System.Collections.Generic.List[object]]::new()
function Add-TrainingFile {
    param([string]$SourcePath, [string]$RelativeDestination)
    $file = Get-Item -LiteralPath $SourcePath -ErrorAction Stop
    if ($file.PSIsContainer -or $file.Length -eq 0) { throw "Missing or empty training file: $SourcePath" }
    $plan.Add([pscustomobject]@{ Source = $file.FullName; RelativeDestination = $RelativeDestination })
}

# Joint-model archives: the manifest must cover all seven conditions and 16 trajectories.
foreach ($name in @('wide16_v3_joint_dataset', 'grid16_legacy_joint_dataset_m0_m4')) {
    $folder = Join-Path $sourceRootPath "data\$name"
    $manifestPath = Join-Path $folder 'manifest.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $pairs = @($manifest.trials | ForEach-Object { "$($_.condition)/$($_.trajectory)" } | Sort-Object -Unique)
    if (@($manifest.conditions).Count -ne 7 -or @($manifest.trials).Count -ne 112 -or $pairs.Count -ne 112) {
        throw "Incomplete seven-condition training archive: $folder"
    }
    $datasetPath = Join-Path $folder 'dataset.npz'
    if ((Get-FileHash -LiteralPath $datasetPath -Algorithm SHA256).Hash -ne $manifest.dataset_sha256) {
        throw "Dataset hash does not match manifest: $datasetPath"
    }
    Add-TrainingFile $manifestPath "$name\manifest.json"
    Add-TrainingFile $datasetPath "$name\dataset.npz"
}

# PKAN exports: only split CSVs and their manifest, never raw trials or ROS bags.
$csvSets = @(
    @{ Name = 'grid16_pkan_csv'; Conditions = @('bare_p0', 'ag145_p0', 'ag145_p05', 'ag145_p10', 'mdpg_p0') },
    @{ Name = 'grid16_fourier_pkan_csv'; Conditions = @('bare_p0', 'ag145_p0', 'ag145_p05', 'ag145_p10', 'mdpg_p0', 'mdpg_p05', 'mdpg_p10') }
)
foreach ($set in $csvSets) {
    $folder = Join-Path $sourceRootPath "data\$($set.Name)"
    $manifestPath = Join-Path $folder 'manifest.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ((@($manifest.conditions.PSObject.Properties.Name | Sort-Object) -join ',') -ne (@($set.Conditions | Sort-Object) -join ',')) {
        throw "Unexpected or missing CSV conditions: $folder"
    }
    Add-TrainingFile $manifestPath "$($set.Name)\manifest.json"
    foreach ($condition in $set.Conditions) {
        foreach ($split in @('train', 'validation', 'test', 'diagnostic')) {
            $relative = "$($set.Name)\$condition\$split.csv"
            Add-TrainingFile (Join-Path $folder "$condition\$split.csv") $relative
        }
    }
}

# Older residual-model datasets live under model/residual_data in the source repo.
foreach ($name in @('rbd_v1', 'dynamics10_v1')) {
    $folder = Join-Path $sourceRootPath "model\residual_data\$name"
    foreach ($filename in @('manifest.json', 'dataset.npz', 'dataset.csv')) {
        Add-TrainingFile (Join-Path $folder $filename) "residual_data\$name\$filename"
    }
}

if ($plan.Count -ne 60) { throw "Expected 60 training files, found $($plan.Count)." }

# Check every source before writing to the destination.
foreach ($item in $plan) {
    $item | Add-Member -NotePropertyName Sha256 -NotePropertyValue (Get-FileHash -LiteralPath $item.Source -Algorithm SHA256).Hash
}

[System.IO.Directory]::CreateDirectory($destinationRootPath) | Out-Null
$expected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$copied = 0
$skipped = 0
$totalBytes = [long]0
foreach ($item in $plan) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $destinationRootPath $item.RelativeDestination))
    if (-not $target.StartsWith($destinationRootPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination path escaped data folder: $target"
    }
    $null = $expected.Add($target)
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
    if ((Test-Path -LiteralPath $target -PathType Leaf) -and
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $item.Sha256) {
        $skipped++
    } else {
        Copy-Item -LiteralPath $item.Source -Destination $target -Force
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $item.Sha256) {
            throw "Copied file hash mismatch: $target"
        }
        $copied++
    }
    $totalBytes += (Get-Item -LiteralPath $target).Length
}

$extras = @(Get-ChildItem -LiteralPath $destinationRootPath -File -Recurse -Force |
    Where-Object { $_.Name -ne '.gitkeep' -and -not $expected.Contains($_.FullName) })
if ($extras.Count) {
    throw "Destination contains $($extras.Count) extra files; no files were deleted. First: $($extras[0].FullName)"
}

[pscustomobject]@{
    TrainingFiles = $plan.Count
    Copied = $copied
    AlreadyMatched = $skipped
    MiB = [math]::Round($totalBytes / 1MB, 2)
    Destination = $destinationRootPath
}
