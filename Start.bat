@echo off
chcp 65001
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo .venv Python not found.
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

echo =====================================================
echo  F5 / F6 설정 대기
echo =====================================================
echo.
echo F5: 좌표 추가
echo F6: 모델, 유사도, 좌표 선택 후 실행
echo F8: 시작 ^| F9: 정지 ^| Ctrl + C: 종료
echo.

"%PYTHON%" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo Python 실행 중 오류가 발생했습니다.
    pause
)
exit /b %errorlevel%
