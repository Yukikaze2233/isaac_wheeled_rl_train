param([string]$Root = 'D:\isaac60-native', [int]$TimeoutSeconds = 3600)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$archive = Join-Path $Root 'downloads\isaac-sim-standalone-6.0.0-windows-x86_64.zip'
$receipt = Get-Content ($archive + '.json') -Raw | ConvertFrom-Json
if ($receipt.status -ne 'verified' -or $receipt.md5 -ne '4b49a4258792f09300ece31be1b6cfd9') {
    throw 'Verified official archive receipt is required'
}
if ((Get-Item $archive).Length -ne 10668464877) { throw 'Archive size changed' }
$destination = Join-Path $Root 'sim'
$partial = Join-Path $Root 'sim.extracting'
if (Test-Path $destination) { throw 'Existing Sim directory will not be overwritten' }
if (Test-Path $partial) { throw 'Inspect the previous extraction before resuming' }
[void](New-Item -ItemType Directory $partial)
$logs = Join-Path $Root 'audits\extract'
[void](New-Item -ItemType Directory -Force $logs)
$process = Start-Process -FilePath "$env:WINDIR\System32\tar.exe" -ArgumentList @('-xf', $archive, '-C', $partial) `
    -RedirectStandardOutput (Join-Path $logs 'tar.stdout.log') -RedirectStandardError (Join-Path $logs 'tar.stderr.log') -PassThru
$nativeHandle = $process.Handle
$record = @{pid=$process.Id; started=[DateTime]::UtcNow.ToString('o'); timeout=$false}
try {
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $record.timeout=$true
        & taskkill.exe /PID $process.Id /T /F
        throw 'Windows extraction exceeded bounded lifetime'
    }
    $record.exit_code=$process.ExitCode
    if ($process.ExitCode -ne 0) { throw "Extraction exited $($process.ExitCode)" }
    foreach ($file in @('isaac-sim.bat','isaac-sim.streaming.bat','python.bat','kit\python\python.exe')) {
        if (-not (Test-Path (Join-Path $partial $file))) { throw "Missing native entrypoint $file" }
    }
    Move-Item $partial $destination
    $record.status='installed'
} finally {
    $record.finished=[DateTime]::UtcNow.ToString('o')
    $record | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $logs 'receipt.json') -Encoding UTF8
}
