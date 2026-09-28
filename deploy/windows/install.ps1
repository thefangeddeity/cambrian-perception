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
    [string]$Source = "0",
    [string]$Checkpoint = "",
    [string]$Model = "",
    [string]$InstallDir = "C:\ProgramData\cambrian\cambrian-perception",
    [int]$ViewerPort = 8090
)
$ErrorActionPreference = "Stop"
$Task = "cambrian-perception"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this in an elevated PowerShell (as Administrator)."
}
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Write-Host "cambrian-perception: installing from $Repo to $InstallDir"

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
if (-not (Test-Path $modelDst)) { Write-Warning "no YOLO model at $modelDst -- it runs without prey (snacks only) until one is added (-Model <yolov8n.onnx>)" }

# 4. What it watches: the camera (it has it whenever it runs: the livecam is off).
Write-Host "  source: $Source"
@{ source = $Source; viewer_port = $ViewerPort } | ConvertTo-Json | Set-Content -Encoding utf8 "$InstallDir\cambrian.json"

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
