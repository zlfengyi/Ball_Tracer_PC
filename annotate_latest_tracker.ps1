param(
    [string]$InputJson = "",
    [string]$OutputDir = "",
    [string]$TrackerConfig = "",
    [string]$RacketModel = "",
    [double]$RacketConf = 0.25,
    [int]$MaxFrames = -1,
    [switch]$NoVideo
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    & chcp.com 65001 > $null
} catch {
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "tracker_output"
}
if ([string]::IsNullOrWhiteSpace($TrackerConfig)) {
    $TrackerConfig = Join-Path $PSScriptRoot "src\config\tracker.json"
}
if ([string]::IsNullOrWhiteSpace($RacketModel)) {
    $RacketModel = Join-Path $PSScriptRoot "yolo_model\racket.onnx"
}

function Get-ToolPython {
    $ros2Python = Join-Path $PSScriptRoot ".venv_ros2\Scripts\python.exe"
    $cleanPython = Join-Path $PSScriptRoot ".venv_clean\Scripts\python.exe"
    foreach ($candidate in @($ros2Python, $cleanPython)) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }
    throw "No Python interpreter found for annotation."
}

function Get-LatestBaseTrackerJson {
    param([string]$SearchDir)

    $latest = Get-ChildItem -Path $SearchDir -File -Filter 'tracker_*.json' |
        Where-Object { $_.BaseName -match '^tracker_\d{8}_\d{6}$' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $latest) {
        throw "No base tracker JSON found under $SearchDir"
    }
    return $latest
}

function Get-TrackerVideoPath {
    param([System.IO.FileInfo]$BaseJson)

    $baseStem = $BaseJson.BaseName
    $baseDir = $BaseJson.DirectoryName
    $candidates = New-Object System.Collections.Generic.List[string]

    try {
        $jsonData = Get-Content -LiteralPath $BaseJson.FullName -Raw | ConvertFrom-Json
        $artifactPath = $jsonData.config.video_output.artifact_path
        if (-not [string]::IsNullOrWhiteSpace($artifactPath)) {
            $candidates.Add($artifactPath)
        }
    } catch {
    }

    foreach ($ext in @(".mp4", ".avi", ".mov", ".mkv")) {
        $candidates.Add((Join-Path $baseDir ($baseStem + $ext)))
    }

    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Tracker video not found for $($BaseJson.FullName)"
}

$python = Get-ToolPython
$annotateScript = Join-Path $PSScriptRoot "test_src\annotate_video.py"
$htmlScript = Join-Path $PSScriptRoot "test_src\generate_curve3_html.py"

if (-not (Test-Path $annotateScript)) {
    throw "Annotate script not found: $annotateScript"
}
if (-not (Test-Path $htmlScript)) {
    throw "HTML generator not found: $htmlScript"
}

$baseJson = if ([string]::IsNullOrWhiteSpace($InputJson)) {
    Get-LatestBaseTrackerJson -SearchDir $OutputDir
} else {
    Get-Item -LiteralPath $InputJson
}

$baseStem = $baseJson.BaseName
$baseDir = $baseJson.DirectoryName
$videoPath = Get-TrackerVideoPath -BaseJson $baseJson
$annotatedVideoPath = Join-Path $baseDir ($baseStem + "_annotated.avi")
$mergedJsonPath = Join-Path $baseDir ($baseStem + "_with_racket.json")
$racketJsonPath = Join-Path $baseDir ($baseStem + "_racket.json")
$htmlPath = Join-Path $baseDir ($baseStem + "_with_racket.html")

$racketConfText = [System.Globalization.CultureInfo]::InvariantCulture.TextInfo.ToLower(
    $RacketConf.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

$annotateArgs = @(
    $annotateScript,
    "--input", $baseJson.FullName,
    "--video", $videoPath,
    "--json-output", $mergedJsonPath,
    "--racket-json-output", $racketJsonPath,
    "--tracker-config", $TrackerConfig,
    "--racket-model", $RacketModel,
    "--racket-conf", $racketConfText
)
if (-not $NoVideo) {
    $annotateArgs += @("--output", $annotatedVideoPath)
} else {
    $annotateArgs += "--no-output-video"
}
if ($MaxFrames -ge 0) {
    $annotateArgs += @("--max-frames", $MaxFrames.ToString())
}

Write-Host "Annotating latest tracker run:"
Write-Host "  Base JSON: $($baseJson.FullName)"
Write-Host "  Video:     $videoPath"
& $python @annotateArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$htmlArgs = @(
    $htmlScript,
    "--input", $baseJson.FullName,
    "--racket-json", $racketJsonPath,
    "--output", $htmlPath
)

& $python @htmlArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done."
if (-not $NoVideo) {
    Write-Host "  Annotated video: $annotatedVideoPath"
}
Write-Host "  Merged JSON:     $mergedJsonPath"
Write-Host "  Racket JSON:     $racketJsonPath"
Write-Host "  HTML:            $htmlPath"
