param(
    [string]$Root = 'D:\isaac60-native',
    [Parameter(Mandatory=$true)][string]$StatePath,
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$RunName,
    [int]$Seconds = 3600,
    [switch]$Minimal
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid run label' }
$os = Get-CimInstance Win32_OperatingSystem
$minimum = if ($Minimal) { 2 } else { 4 }
if ($os.FreePhysicalMemory / 1MB -lt $minimum) { throw "Renderer needs at least $minimum GiB available host RAM" }
foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','PYTHONHOME','PYTHONPATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:TMP = $env:TEMP = Join-Path $Root 'tmp'
$env:CUDA_CACHE_PATH = Join-Path $Root 'cache\cuda'
$env:PYTHONNOUSERSITE = '1'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$run = Join-Path $Root ('audits\' + $RunName)
$env:ISAAC_NATIVE_RUN_DIR = $run
$env:V5_MIRROR_STATE = $StatePath
$env:V5_MIRROR_BUNDLE = $Bundle
$entry = if ($Minimal) { 'call "' + (Join-Path $Root 'sim\kit\kit.exe') + '" "' + (Join-Path $Root 'sim\apps\v5.training.mirror.kit') + '"' } else { 'call isaac-sim.streaming.bat' }
$command = $entry + ' --no-window --portable --portable-root "' + (Join-Path $Root 'portable') + '"' +
    ' --/app/settings/persistent=false --/app/file/ignoreUnsavedOnExit=true --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0' +
    ' --/app/window/hideUi=false --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720' +
    ' --/app/runLoops/main/rateLimitEnabled=true --/app/runLoops/main/rateLimitFrequency=30' +
    ' --/exts/omni.kit.livestream.app/primaryStream/publicIp=127.0.0.1' +
    ' --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998' +
    ' --/exts/omni.services.livestream.session/quitOnSessionEnded=false' +
    ' --/log/file="' + (Join-Path $run 'kit.log') + '" --exec "' + (Join-Path $PSScriptRoot 'v5_state_gui.py') + '"'
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory (Join-Path $Root 'sim') -OutputDirectory $run -Seconds $Seconds
exit $LASTEXITCODE
