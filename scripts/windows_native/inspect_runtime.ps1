param([string]$Root='D:\isaac60-native', [string]$Label='runtime-before', [switch]$Cuda)
$ErrorActionPreference='Stop'
foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','PYTHONHOME','PYTHONPATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:PYTHONNOUSERSITE='1'
$command='call python.bat "'+(Join-Path $Root 'tools\inspect_runtime.py')+'" --output "'+(Join-Path $Root ('audits\'+$Label+'.json'))+'"'
if ($Cuda) { $command+=' --cuda' }
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory (Join-Path $Root 'sim') `
    -OutputDirectory (Join-Path $Root ('audits\'+$Label)) -Seconds 90
exit $LASTEXITCODE
