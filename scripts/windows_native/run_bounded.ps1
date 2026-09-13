# Own one Windows process tree, persist evidence, and bound its lifetime independently of WSL.
param(
    [Parameter(Mandatory=$true)][string]$CommandLine,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Seconds = 900
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if ($Seconds -lt 1 -or $Seconds -gt 14400) { throw 'Bound must be 1..14400 seconds' }
if (Test-Path $OutputDirectory) { throw "Refusing an existing run directory: $OutputDirectory" }
[void](New-Item -ItemType Directory -Path $OutputDirectory)
$record = @{started=[DateTime]::UtcNow.ToString('o'); command=$CommandLine; timeout=$false}
$record | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'started.json')
$process = $null
$stdoutFile = $stderrFile = $stdoutCopy = $stderrCopy = $null
try {
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $env:ComSpec
    $info.Arguments = '/d /s /c ' + $CommandLine
    $info.WorkingDirectory = $WorkingDirectory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    [void]$process.Start()
    # Keep the actual started Process handle; Start-Process -PassThru can lose ExitCode.
    $stdoutFile = New-Object System.IO.FileStream((Join-Path $OutputDirectory 'stdout.log'),
        [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1)
    $stderrFile = New-Object System.IO.FileStream((Join-Path $OutputDirectory 'stderr.log'),
        [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1)
    $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdoutFile)
    $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderrFile)
    # Retain the native process handle before it exits; PowerShell 5 may otherwise
    # return a null ExitCode from a Start-Process/WaitForExit combination.
    $nativeHandle = $process.Handle
    $record.pid = $process.Id
    $record.session_id = $process.SessionId
    $record | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'running.json')
    if (-not $process.WaitForExit($Seconds * 1000)) {
        $record.timeout = $true
        & taskkill.exe /PID $process.Id /T /F | Out-File -Encoding UTF8 (Join-Path $OutputDirectory 'timeout-kill.log')
        $process.WaitForExit()
    }
    $record.exit_code = $process.ExitCode
    if ($null -eq $record.exit_code) { throw 'Windows process exited without an observable exit code' }
} catch {
    $record.error = $_.Exception.Message
    throw
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        & taskkill.exe /PID $process.Id /T /F | Out-File -Encoding UTF8 (Join-Path $OutputDirectory 'cleanup-kill.log')
        $process.WaitForExit()
    }
    $record.finished = [DateTime]::UtcNow.ToString('o')
    $record.process_exited = ($null -eq $process -or $process.HasExited)
    if ($null -ne $stdoutCopy) { [void]$stdoutCopy.Wait(5000) }
    if ($null -ne $stderrCopy) { [void]$stderrCopy.Wait(5000) }
    if ($null -ne $stdoutFile) { $stdoutFile.Dispose() }
    if ($null -ne $stderrFile) { $stderrFile.Dispose() }
    $record | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'finished.json')
}
if ($record.timeout) { exit 124 }
exit $record.exit_code
