<#
  Installs (or updates) cambrian-perception on Windows 11 as a background
  service (docs/packaging.md). Run in an elevated PowerShell from the repo:

    powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 [-Checkpoint <path>] [-Source auto|0|rtsp://...]

  Beside laptop-livecam (hls-livecam-win) it never clashes: the two are a
  camera suite and never run together (docs/suite.md). Installing is
  starting: the organism takes the camera, and the livecam stops and stays
  off until `camdash --start`. Port 8090 only, its own task, folders and
  Python venv. Re-running it updates the code and keeps state\ (checkpoint, logs).
#>
param(
    [string]$Source = "",
    [string]$Checkpoint = "",
    [string]$Model = "",
    [string]$InstallDir = "C:\ProgramData\cambrian\cambrian-perception",
    [int]$ViewerPort = 8090,
    # -Hive joins the hive (after an extinction it may take a migrant from its
    # peers, and it serves its own organism to them); a new install is solo
    # (-NoHive), and an update keeps what the host had.
    [switch]$Hive,
    [switch]$NoHive
)
$ErrorActionPreference = "Stop"
$Task = "cambrian-perception"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this in an elevated PowerShell (as Administrator)."
}
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Write-Host "cambrian-perception: installing from $Repo to $InstallDir"

# 0. Its self-test on the incoming code (tools\selftest.py), with the Python
#    already installed, before anything running is touched: failing, nothing changes.
if (($Repo -ne $InstallDir) -and (Test-Path "$InstallDir\.venv\Scripts\python.exe")) {
    $log = Join-Path $env:TEMP "cambrian-selftest.log"
    $keep = $ErrorActionPreference; $ErrorActionPreference = "Continue"  # its stderr is not a failure: its exit code is
    & "$InstallDir\.venv\Scripts\python.exe" "$Repo\tools\selftest.py" *> $log
    $code = $LASTEXITCODE; $ErrorActionPreference = $keep
    if ($code -ne 0) {
        Get-Content $log -Tail 20
        throw "self-test FAILED -- the running organism and its code are unchanged; see $log"
    }
    Write-Host "  self-test passed"
}

# 1. Stop a running copy (it saves its checkpoint), then lay down the code.
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*cambrian_service.py*" -and $_.Name -like "python*" }
if ($running) {
    Write-Host "  stopping the running organism (it saves first)..."
    New-Item -ItemType File -Force "$InstallDir\state\service.stop" | Out-Null
    for ($i = 0; $i -lt 200 -and (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*cambrian_service.py*" }); $i++) { Start-Sleep 1 }
}
if ($Repo -ne $InstallDir) {
    robocopy $Repo $InstallDir /E /XD .venv .git state __pycache__ /XF cambrian.json /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying the code failed (robocopy $LASTEXITCODE)" }
}
New-Item -ItemType Directory -Force "$InstallDir\state", "$InstallDir\models" | Out-Null
# `cambrian --stop | --yield` leave their notes in state\ from any terminal, elevated or not.
icacls "$InstallDir\state" /grant "*S-1-5-32-545:(OI)(CI)M" /Q | Out-Null  # BUILTIN\Users: modify

# 2. Python (3.11+), then the organism's own venv with the pinned libraries.
function Find-Python {
    foreach ($c in @("py -3.13", "py -3.12", "py -3.14", "py -3", "python")) {
        $exe, $arg = $c.Split(" ", 2)
        try {
            $v = & $exe $arg -c "import sys; print(sys.executable if sys.version_info >= (3, 11) else '')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
        } catch {}
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    Write-Host "  no Python 3.11+ found -- installing Python 3.13 with winget"
    winget install -e --id Python.Python.3.13 --scope machine --silent --accept-package-agreements --accept-source-agreements | Out-Null
    $py = "C:\Program Files\Python313\python.exe"
    if (-not (Test-Path $py)) { throw "Python install failed; install Python 3.12+ and re-run." }
}
Write-Host "  python: $py"
if (-not (Test-Path "$InstallDir\.venv\Scripts\python.exe")) { & $py -m venv "$InstallDir\.venv" }
& "$InstallDir\.venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r "$InstallDir\requirements.lock"
if ($LASTEXITCODE -ne 0) { throw "installing the pinned libraries (requirements.lock) failed" }

# 3. The prey detector's model.
$modelDst = "$InstallDir\models\yolov8n.onnx"
$livecamModel = "C:\Program Files\hls-livecam-win\models\yolov8n.onnx"
if ($Model) { Copy-Item $Model $modelDst -Force }
elseif (-not (Test-Path $modelDst) -and (Test-Path $livecamModel)) { Copy-Item $livecamModel $modelDst -Force }
# the rest (or all of them) from the release, checksummed (tools\fetch_models.py)
& "$InstallDir\.venv\Scripts\python.exe" "$InstallDir\tools\fetch_models.py" "$InstallDir\models"

# 4. What it watches: -Source, else what it watched before (an update keeps
#    it), else the camera (it has it whenever it runs: the livecam is off).
if (-not $Source -and (Test-Path "$InstallDir\cambrian.json")) {
    try { $Source = [string](Get-Content "$InstallDir\cambrian.json" -Raw | ConvertFrom-Json).source } catch {}
}
if (-not $Source) { $Source = "0" }
Write-Host "  source: $Source"
$hiveArg = if ($Hive) { "true" } elseif ($NoHive) { "false" } else { "keep" }  # (Windows PowerShell drops an empty argument)
& "$InstallDir\.venv\Scripts\python.exe" "$InstallDir\tools\settings.py" "$InstallDir\cambrian.json" $hiveArg "source=$Source" "viewer_port=$ViewerPort"
if ($LASTEXITCODE -ne 0) { throw "writing cambrian.json failed" }

# 5. A lineage to continue, if given and none is here yet.
if ($Checkpoint -and -not (Test-Path "$InstallDir\state\checkpoint.json")) {
    Copy-Item $Checkpoint "$InstallDir\state\checkpoint.json"
    Write-Host "  seeded the organism from $Checkpoint"
}

# 6. The viewer's port, on private networks (home, Tailscale).
if (-not (Get-NetFirewallRule -DisplayName "cambrian-perception viewer" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "cambrian-perception viewer" -Direction Inbound -Protocol TCP -LocalPort $ViewerPort `
        -Action Allow -Profile Private, Domain | Out-Null
}

# 7. The boot task: the supervisor, as this user whether logged in or not (S4U, no
#    stored password), least privilege, no run-time limit, on battery, restarting.
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute "$InstallDir\.venv\Scripts\pythonw.exe" -Argument "tools\cambrian_service.py" -WorkingDirectory $InstallDir
$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $Task -Action $action -Trigger (New-ScheduledTaskTrigger -AtStartup) -Principal $principal `
    -Settings $settings -Force | Out-Null

# 8. `cambrian` on the PATH.
$bin = "$InstallDir\deploy\windows"
$path = [Environment]::GetEnvironmentVariable("Path", "Machine")
if (($path -split ";") -notcontains $bin) { [Environment]::SetEnvironmentVariable("Path", "$path;$bin", "Machine") }

# 9. Start: it takes the camera (the livecam yields).
& "$InstallDir\.venv\Scripts\python.exe" "$InstallDir\tools\cambrian_ctl.py" --start
Write-Host "installed. Control it with: cambrian --start | --stop | --restart | --yield | --status (a new terminal picks up the PATH)"
