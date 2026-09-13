param([string]$Root='D:\isaac60-native', [Parameter(Mandatory=$true)][string]$RunName,
      [string]$Reason='operator_stop')
$ErrorActionPreference='Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid run name' }
$run=Join-Path $Root ('audits\'+$RunName)
$record=Get-Content (Join-Path $run 'running.json') -Raw | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$record.pid)
if ($null -eq $process) { return }
if ($process.Name -ne 'cmd.exe' -or -not $process.CommandLine.TrimEnd().EndsWith(
    $record.command.TrimEnd(), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'PID command no longer matches this owned run'
}
$result=@{pid=$record.pid;reason=$Reason;requested=[DateTime]::UtcNow.ToString('o')}
& taskkill.exe /PID $record.pid /T /F
$result.exit_code=$LASTEXITCODE
$result | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $run 'operator-stop.json')
exit $LASTEXITCODE
