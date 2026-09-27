@echo off
rem cambrian --start | --stop | --restart | --status  (see tools\cambrian_ctl.py)
"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0..\..\tools\cambrian_ctl.py" %*
