# Preparation by default. Only the coordinating agent should opt in after resource scheduling.
param(
    [string]$Root='D:\isaac60-native',
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$CheckpointSha256,
    [Parameter(Mandatory=$true)][string]$SourceRun,
    [string]$RunName=('checkpoint-replay-'+(Get-Date -Format 'yyyyMMddTHHmmss')),
    [int]$Seconds=90,
    [double]$MinimumFreeGiB=12,
    [switch]$Launch
)
$ErrorActionPreference='Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$' -or $CheckpointSha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid run/hash' }
if ($Seconds -lt 1 -or $Seconds -gt 300) { throw 'Replay bound must be 1..300 seconds' }
$os=Get-CimInstance Win32_OperatingSystem
$freeGiB=$os.FreePhysicalMemory/1MB
$run=Join-Path $Root ('audits\'+$RunName)
$output=Join-Path $run 'replay'
$command='call "'+(Join-Path $Root 'sim\python.bat')+'" "'+(Join-Path $Root 'tools\checkpoint_replay.py')+'"'+
    ' --native-root "'+$Root+'" --repo "'+$Repo+'" --checkpoint "'+$Checkpoint+'"'+
    ' --checkpoint-sha256 '+$CheckpointSha256+' --source-run "'+$SourceRun+'"'+
    ' --output "'+$output+'" --seconds '+$Seconds+' --launch'
@{classification='TRAINING CHECKPOINT REPLAY';launch_requested=[bool]$Launch;free_ram_GiB=$freeGiB;
  required_free_GiB=$MinimumFreeGiB;command=$command;current_training_state=$false} | ConvertTo-Json
if (-not $Launch) { return }
if ($freeGiB -lt $MinimumFreeGiB) { throw 'Insufficient current host headroom; coordinate training/benchmark first' }
foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','PYTHONHOME','PYTHONPATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:PYTHONNOUSERSITE='1'
$env:TMP=$env:TEMP=Join-Path $Root 'tmp'
$env:OMNI_KIT_ACCEPT_EULA='YES'
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory $Repo `
    -OutputDirectory $run -Seconds ($Seconds+600)
exit $LASTEXITCODE
