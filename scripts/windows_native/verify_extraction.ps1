param([string]$Root='D:\isaac60-native')
$ErrorActionPreference='Stop'
$python='D:\Python\Python3.11.4\python.exe'
$command='call "'+$python+'" -I -B "'+(Join-Path $Root 'tools\verify_extraction.py')+'"' +
    ' --archive "'+(Join-Path $Root 'downloads\isaac-sim-standalone-6.0.0-windows-x86_64.zip')+'"' +
    ' --directory "'+(Join-Path $Root 'sim.extracting')+'" --publish "'+(Join-Path $Root 'sim')+'"' +
    ' --receipt "'+(Join-Path $Root 'audits\extract-verification.json')+'"'
& (Join-Path $Root 'tools\run_bounded.ps1') -CommandLine $command -WorkingDirectory $Root `
    -OutputDirectory (Join-Path $Root 'audits\verify-extraction') -Seconds 900
exit $LASTEXITCODE
