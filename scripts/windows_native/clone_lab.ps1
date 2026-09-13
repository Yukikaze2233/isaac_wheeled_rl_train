param([string]$Root='D:\isaac60-native')
$ErrorActionPreference='Stop'
$destination=Join-Path $Root 'source\IsaacLab'
if (Test-Path $destination) { throw 'Existing source checkout will not be replaced' }
[void](New-Item -ItemType Directory -Force (Join-Path $Root 'source'))
$git=(Get-Command git.exe -ErrorAction Stop).Source
$command='call "'+$git+'" -c core.autocrlf=false clone --depth 1 --single-branch --branch v3.0.0-beta2.patch1 https://github.com/isaac-sim/IsaacLab.git "'+$destination+'"'
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory $Root `
    -OutputDirectory (Join-Path $Root 'audits\clone-lab') -Seconds 300
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$head=& $git -C $destination rev-parse HEAD
if ($head -ne 'ffff603eafc6b74264a5261cc0183d6a65390d78') { throw "Unexpected Lab commit: $head" }
$origin=& $git -C $destination remote get-url origin
$status=& $git -C $destination status --porcelain
if ($status) { throw 'Fresh Lab checkout is dirty' }
@{commit=$head;tag='v3.0.0-beta2.patch1';origin=$origin;clean=$true} | ConvertTo-Json |
    Set-Content -Encoding UTF8 (Join-Path $Root 'audits\lab-source.json')
