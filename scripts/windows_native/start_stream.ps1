param(
    [string]$Root = 'D:\isaac60-native',
    [Parameter(Mandatory=$true)][string]$ServerAddress,
    [string]$RunName = ('stream-' + (Get-Date -Format 'yyyyMMddTHHmmss')),
    [int]$Seconds = 900,
    [switch]$Probe
)
$ErrorActionPreference = 'Stop'
if ($ServerAddress -notmatch '^[A-Za-z0-9.:-]+$') { throw 'Invalid advertised server address' }
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must be a unique simple label' }
$sim = Join-Path $Root 'sim'
if (-not (Test-Path (Join-Path $sim 'isaac-sim.streaming.bat'))) { throw 'Native Sim not installed' }
foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','PYTHONHOME','PYTHONPATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
foreach ($name in @('tmp','cache','portable','cache\cuda')) {
    [void](New-Item -ItemType Directory -Force (Join-Path $Root $name))
}
$env:TMP = $env:TEMP = Join-Path $Root 'tmp'
$env:CUDA_CACHE_PATH = Join-Path $Root 'cache\cuda'
$env:PYTHONNOUSERSITE = '1'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$run = Join-Path $Root ('audits\' + $RunName)
$env:ISAAC_NATIVE_RUN_DIR = $run
$command = 'call isaac-sim.streaming.bat --no-window --portable --portable-root "' + (Join-Path $Root 'portable') + '"' +
    ' --/app/settings/persistent=false --/app/file/ignoreUnsavedOnExit=true --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0' +
    ' --/app/window/hideUi=false --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720' +
    ' --/exts/omni.kit.livestream.app/primaryStream/publicIp=' + $ServerAddress +
    ' --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998' +
    ' --/exts/omni.services.livestream.session/quitOnSessionEnded=false' +
    ' --/log/file="' + (Join-Path $run 'kit.log') + '"'
if ($Probe) { $command += ' --exec "' + (Join-Path $Root 'tools\gui_probe.py') + '"' }
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory $sim -OutputDirectory $run -Seconds $Seconds
exit $LASTEXITCODE
