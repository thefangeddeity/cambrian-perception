<#
  Removes cambrian-perception's service from Windows. Keeps state\ (the
  organism's checkpoint and logs) unless -Purge. Run elevated.
#>
param([string]$InstallDir = "C:\ProgramData\cambrian\cambrian-perception", [switch]$Purge)
$ErrorActionPreference = "Continue"
if (Test-Path "$InstallDir\.venv\Scripts\python.exe") {
    & "$InstallDir\.venv\Scripts\python.exe" "$InstallDir\tools\cambrian_ctl.py" --stop
}
Unregister-ScheduledTask -TaskName "cambrian-perception" -Confirm:$false -ErrorAction SilentlyContinue
Get-NetFirewallRule -DisplayName "cambrian-perception viewer" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
$bin = "$InstallDir\deploy\windows"
$path = [Environment]::GetEnvironmentVariable("Path", "Machine")
[Environment]::SetEnvironmentVariable("Path", (($path -split ";") | Where-Object { $_ -and $_ -ne $bin }) -join ";", "Machine")
if ($Purge) {
    Remove-Item -Recurse -Force $InstallDir
    Write-Host "removed, including the organism's state"
} else {
    Get-ChildItem $InstallDir -Exclude state | Remove-Item -Recurse -Force
    Write-Host "removed; the organism's state is kept in $InstallDir\state"
}
