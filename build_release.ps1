<#
.SYNOPSIS
    Builds a self-contained Windows release of VOD BLADE: embeddable Python +
    all pip dependencies + bundled binaries (ffmpeg, TwitchDownloaderCLI,
    YAMNet model) + a packaged launcher, zipped for a GitHub Release.

.DESCRIPTION
    Only ever reads from the repo root and writes to .\dist\ - never touches
    any tracked source file. Safe to re-run any time; delete .\dist\ to start
    clean.

.PARAMETER Version
    Overrides the version written into the release's VERSION file. Defaults
    to whatever's already in .\VERSION.

.PARAMETER VendorPath
    Folder containing binaries this repo doesn't check into git:
    TwitchDownloaderCLI.exe, models\yamnet.onnx, models\yamnet_class_map.csv.
    Defaults to E:\vod-blade-vendor\bin - maintained by hand outside the repo,
    since there's no confirmed stable "latest" download URL for
    TwitchDownloaderCLI the way there is for Ollama/ffmpeg.

.EXAMPLE
    .\build_release.ps1
    .\build_release.ps1 -Version 0.2.0 -VendorPath D:\my-vendor-bin
#>
param(
    [string]$Version,
    [string]$VendorPath = "E:\vod-blade-vendor\bin"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest's default progress bar is very slow for big files

# Without this, pip install can silently pick up packages from the dev machine's
# OTHER Python 3.14 install's user-site directory (%APPDATA%\Roaming\Python\...) once
# `import site` is enabled in the embeddable distro's ._pth file below - considering a
# requirement "already satisfied" from there instead of installing a real local copy.
# That defeats the whole point of a self-contained, portable build.
$env:PYTHONNOUSERSITE = "1"

$RepoRoot = $PSScriptRoot
$DistRoot = Join-Path $RepoRoot "dist"
$CacheDir = Join-Path $DistRoot "_cache"       # downloaded archives, reused across builds
$StageDir = Join-Path $DistRoot "vod-blade-build"

# --- Guard rail: refuse to touch anything outside dist/, no matter what the
# --- params/cwd end up being. This is the actual answer to "how hard is it to
# --- redo if we mess up the build" - the blast radius of a bad run is
# --- contained to a folder that's safe to delete and rebuild from scratch.
function Assert-UnderDist([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $distResolved = [System.IO.Path]::GetFullPath($DistRoot)
    if (-not $resolved.StartsWith($distResolved, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to touch '$resolved' - it's outside dist/ ('$distResolved')."
    }
}

if (-not $Version) {
    $versionFile = Join-Path $RepoRoot "VERSION"
    if (-not (Test-Path $versionFile)) { throw "No -Version given and no VERSION file found at $versionFile." }
    $Version = (Get-Content $versionFile -Raw).Trim()
}
Write-Host "[build] Version: $Version"

# --- 1. Clean/recreate the staging folder only ---
Assert-UnderDist $StageDir
if (Test-Path $StageDir) { Remove-Item $StageDir -Recurse -Force }
New-Item -ItemType Directory -Path $StageDir | Out-Null
New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null

# --- 2. Fetch embeddable Python (pinned to match the dev .venv's interpreter) ---
$PyVersion = "3.14.5"
$PyTag = "python314"  # matches the ._pth filename Python's embeddable zip ships for 3.14.x
$PyZipUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$PyZipPath = Join-Path $CacheDir "python-$PyVersion-embed-amd64.zip"
$PyDir = Join-Path $StageDir "python"

if (-not (Test-Path $PyZipPath)) {
    Write-Host "[build] Downloading embeddable Python $PyVersion ..."
    Invoke-WebRequest -Uri $PyZipUrl -OutFile $PyZipPath
} else {
    Write-Host "[build] Using cached embeddable Python zip."
}
Write-Host "[build] Extracting embeddable Python ..."
Expand-Archive -Path $PyZipPath -DestinationPath $PyDir -Force

# --- 3. Enable pip in the embeddable distro, and let it find the app's own source ---
# Order matters: the ._pth edit must land BEFORE get-pip.py runs, or pip's own
# install can't find the `site` module it needs.
#
# The embeddable distribution's ._pth file puts the interpreter in an isolated path
# mode that does NOT do the normal "add the running script's own directory to
# sys.path" - so without the appended ".." line below (relative to python\, i.e. the
# release root where app.py/config.py/core/ live), `python.exe ..\app.py` starts fine
# but immediately fails with "ModuleNotFoundError: No module named 'config'" the
# moment it tries to import a sibling module. Confirmed by an actual smoke-test run.
$PthFile = Join-Path $PyDir "$PyTag._pth"
(Get-Content $PthFile) -replace '^#\s*import site', 'import site' | Set-Content $PthFile
Add-Content -Path $PthFile -Value ".."

$GetPipPath = Join-Path $CacheDir "get-pip.py"
if (-not (Test-Path $GetPipPath)) {
    Write-Host "[build] Downloading get-pip.py ..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPipPath
}
$PyExe = Join-Path $PyDir "python.exe"
Write-Host "[build] Bootstrapping pip ..."
& $PyExe $GetPipPath --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed with exit code $LASTEXITCODE" }

# Modern get-pip.py no longer bundles setuptools/wheel, but some deps (e.g. srt)
# only ship a source distribution and need setuptools's build_meta backend to
# build - without this, pip install fails with "Cannot import 'setuptools.build_meta'".
Write-Host "[build] Installing setuptools/wheel (needed to build source-only deps) ..."
& $PyExe -m pip install --no-warn-script-location setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip install setuptools/wheel failed with exit code $LASTEXITCODE" }

# --- 4. Install dependencies ---
# --no-build-isolation: pip's normal build isolation creates an ephemeral venv per
# sdist build, which the embeddable distribution can't support (it ships without the
# `venv` module). Building directly against the environment we just put
# setuptools/wheel into (the standard workaround for embeddable-Python packaging)
# avoids that entirely. Safe here since every dependency either ships a prebuilt
# wheel already or (like srt) is a small pure-Python sdist with no exotic build deps.
Write-Host "[build] Installing requirements.txt ..."
& $PyExe -m pip install --no-warn-script-location --no-build-isolation -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }

# --- 5. Copy app source (explicit allowlist, never a blanket copy - avoids
# --- shipping the dev's .env, data/, .venv/, __pycache__) ---
Write-Host "[build] Copying app source ..."
$SourceItems = @("app.py", "config.py", "VERSION", "core", "ui", "exporters", "utils")
foreach ($item in $SourceItems) {
    $src = Join-Path $RepoRoot $item
    $dst = Join-Path $StageDir $item
    if (Test-Path $src -PathType Container) {
        # robocopy exit codes 0-7 are all success; /E copies subfolders incl. empty ones
        robocopy $src $dst /E /XD "__pycache__" /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed copying $item (exit code $LASTEXITCODE)" }
    } else {
        Copy-Item $src $dst
    }
}

# --- 6. Bundle binaries ---
$BinDir = Join-Path $StageDir "bin"
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $BinDir "models") -Force | Out-Null

# ffmpeg/ffprobe - static build from BtbN's rolling "latest" release
$FfmpegZipPath = Join-Path $CacheDir "ffmpeg-win64-gpl.zip"
if (-not (Test-Path $FfmpegZipPath)) {
    Write-Host "[build] Downloading ffmpeg (BtbN static build) ..."
    Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" -OutFile $FfmpegZipPath
} else {
    Write-Host "[build] Using cached ffmpeg zip."
}
$FfmpegExtractDir = Join-Path $CacheDir "ffmpeg-extracted"
if (-not (Test-Path $FfmpegExtractDir)) {
    Expand-Archive -Path $FfmpegZipPath -DestinationPath $FfmpegExtractDir -Force
}
$FfmpegBin = Get-ChildItem -Path $FfmpegExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$FfprobeBin = Get-ChildItem -Path $FfmpegExtractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $FfmpegBin -or -not $FfprobeBin) { throw "Could not find ffmpeg.exe/ffprobe.exe inside the downloaded archive - BtbN's layout may have changed." }
Copy-Item $FfmpegBin.FullName (Join-Path $BinDir "ffmpeg.exe")
Copy-Item $FfprobeBin.FullName (Join-Path $BinDir "ffprobe.exe")

# TwitchDownloaderCLI.exe + YAMNet model - not fetchable from a confirmed stable
# URL, so sourced from a manually-maintained local vendor folder.
if (-not (Test-Path $VendorPath)) {
    throw "VendorPath '$VendorPath' not found. It must contain TwitchDownloaderCLI.exe and models\yamnet.onnx / models\yamnet_class_map.csv. Pass -VendorPath to point elsewhere."
}
$VendorFiles = @(
    "TwitchDownloaderCLI.exe",
    "models\yamnet.onnx",
    "models\yamnet_class_map.csv"
)
foreach ($rel in $VendorFiles) {
    $src = Join-Path $VendorPath $rel
    $dst = Join-Path $BinDir $rel
    if (-not (Test-Path $src)) { throw "Missing vendor file: $src" }
    New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
    Copy-Item $src $dst -Force
}

# --- 7. Write VERSION (overwrite with the resolved -Version, in case it
# --- differs from the source file's) ---
Set-Content -Path (Join-Path $StageDir "VERSION") -Value $Version -NoNewline

# --- 8. Packaged launcher, from the checked-in template (not generated inline,
# --- so its content is reviewable/git-tracked) ---
Copy-Item (Join-Path $RepoRoot "run_app.release.bat.template") (Join-Path $StageDir "run_app.bat")

# --- 9. Zip it up ---
$ZipPath = Join-Path $DistRoot "VOD-BLADE-v$Version-win64.zip"
Assert-UnderDist $ZipPath
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Write-Host "[build] Zipping to $ZipPath ..."
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath

Write-Host "[build] Done: $ZipPath"
