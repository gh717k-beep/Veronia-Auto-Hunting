@echo off
setlocal
cd /d "%~dp0"

set "PS_SCRIPT=%~dp0train_custom_model.ps1"

if not exist "%PS_SCRIPT%" (
  echo.
  echo PS1 file not found:
  echo %PS_SCRIPT%
  echo.
  pause
  exit /b 1
)

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"
exit /b %ERRORLEVEL%