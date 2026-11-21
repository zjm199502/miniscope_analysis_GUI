@echo off
setlocal
chcp 65001 >nul
REM 打包成 Windows 程序（在 Windows 上、本仓库根目录运行）
REM 结果为 dist\CalciLume-Windows-Setup.exe 完整安装包

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .build-venv || goto :error
call .build-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install ".[cuda12]" pyinstaller || goto :error
python -c "import miniscope_gui; open('base_version.txt','w').write(miniscope_gui.__version__)" || goto :error
REM Keep unrelated DLLs on the workstation PATH out of the installer.
for /f "delims=" %%B in ('python -c "import sys; print(sys.base_prefix)"') do set "BUILD_PYTHON=%%B"
set "PATH=%CD%\.build-venv\Scripts;%SystemRoot%\System32;%SystemRoot%;%BUILD_PYTHON%;%BUILD_PYTHON%\DLLs"
pyinstaller --noconfirm --clean --windowed --name MiniscopeAnalysis --icon packaging/calcilume.ico ^
  --add-data "base_version.txt:." --collect-submodules miniscope_gui --collect-data miniscope_gui ^
  --hidden-import cupy --additional-hooks-dir packaging/hooks ^
  --collect-submodules pyqtgraph --collect-data pyqtgraph ^
  --exclude-module numba --exclude-module llvmlite --exclude-module matplotlib ^
  --exclude-module tkinter ^
  packaging\launcher.py || goto :error
echo.
set /p APP_VERSION=<base_version.txt
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File packaging\test_windows_cuda.ps1 || goto :error
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File packaging\build_setup.ps1 -Version "%APP_VERSION%" || goto :error
echo 打包完成：dist\CalciLume-Windows-Setup.exe
goto :eof
:error
echo 打包失败，请把上面的报错信息发给维护者。
exit /b 1
