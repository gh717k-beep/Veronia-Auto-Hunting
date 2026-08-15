@echo off
chcp 65001
cd /d "%~dp0"
setlocal EnableDelayedExpansion
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo .venv Python not found.
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

echo =====================================================
echo  사용할 모델 선택
echo =====================================================
echo.

set /a MODEL_COUNT=0
if exist "%~dp0runs" (
    for /d %%D in ("%~dp0runs\*") do (
        set /a MODEL_COUNT+=1
        set "MODEL_!MODEL_COUNT!=%%~nD"
        echo  [!MODEL_COUNT!] %%~nD
    )
) else (
    echo  - 없음
)

echo.
set /p MODEL_INPUT=모델 번호 또는 이름 입력 (엔터: 최신 모델 자동 선택): 
set /p CONFIDENCE=감지 신뢰도 입력 (0.0 ~ 1.0, 예: 0.55): 

if "%CONFIDENCE%"=="" set "CONFIDENCE=0.7"
set "MODEL_NAME="
if not "%MODEL_INPUT%"=="" (
    set "MODEL_NAME=%MODEL_INPUT%"
    if "%MODEL_INPUT%" neq "" (
        for /l %%I in (1,1,%MODEL_COUNT%) do (
            if /i "%MODEL_INPUT%"=="%%I" set "MODEL_NAME=!MODEL_%%I!"
        )
    )
)

if not "%MODEL_NAME%"=="" (
    echo.
    echo 선택한 모델: %MODEL_NAME%
    echo 신뢰도 기준: %CONFIDENCE%
    echo.
    cmd /k ""%PYTHON%" "%~dp0main.py" "%MODEL_NAME%" "%CONFIDENCE%""
) else (
    echo.
    echo 최신 모델 자동 선택
    echo 신뢰도 기준: %CONFIDENCE%
    echo.
    cmd /k ""%PYTHON%" "%~dp0main.py" "%CONFIDENCE%""
)
rem F8 = 시작, F9 = 정지, Ctrl + C = 종료
