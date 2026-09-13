param([string]$Root='D:\isaac60-native', [string]$RunName='lab-install')
$ErrorActionPreference='Stop'
$lab=Join-Path $Root 'source\IsaacLab'
$sim=Join-Path $Root 'sim'
$git=(Get-Command git.exe -ErrorAction Stop).Source
$head=& $git -C $lab rev-parse HEAD
if ($head -ne 'ffff603eafc6b74264a5261cc0183d6a65390d78') { throw 'Unexpected official Lab source identity' }
$link=Join-Path $lab '_isaac_sim'
if (-not (Test-Path $link)) { [void](New-Item -ItemType Junction -Path $link -Target $sim) }
if ((Get-Item $link).Target -ne $sim) { throw 'Unexpected _isaac_sim target' }
foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','PYTHONHOME','PYTHONPATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:PYTHONNOUSERSITE='1'
$env:UV_CACHE_DIR=Join-Path $Root 'cache\uv'
$env:PIP_CACHE_DIR=Join-Path $Root 'cache\pip'
$env:TEMP=$env:TMP=Join-Path $Root 'tmp'
$env:OMNI_KIT_ACCEPT_EULA='YES'
$uv=(Get-Command uv.exe -ErrorAction Stop).Source
$python=Join-Path $sim 'kit\python\kit.exe'
if (-not (Test-Path $python)) { throw 'Run the bundled Python inspection first' }
$command='call "'+$uv+'" pip install --python "'+$python+'" torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128' +
    ' && call "'+(Join-Path $sim 'python.bat')+'" "'+(Join-Path $Root 'tools\install_lab_core.py')+'"' +
    ' && call "'+$uv+'" pip install --python "'+$python+'" rsl-rl-lib==5.5.1 onnx==1.23.0rc1 onnxruntime==1.30.0 onnxscript'
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory $lab `
    -OutputDirectory (Join-Path $Root ('audits\'+$RunName)) -Seconds 3600
exit $LASTEXITCODE
