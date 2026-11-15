@echo off
chcp 65001 >nul
REM 打包成 Windows 程序（在 Windows 上、本仓库根目录运行）
REM 结果在 dist\MiniscopeAnalysis\MiniscopeAnalysis.exe ，整个文件夹可以拷给同学直接用

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .build-venv || goto :error
call .build-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install . pyinstaller || goto :error
pyinstaller --noconfirm --clean --windowed --name MiniscopeAnalysis ^
  --collect-submodules pyqtgraph --collect-data pyqtgraph ^
  --exclude-module numba --exclude-module llvmlite --exclude-module matplotlib ^
  --exclude-module tkinter ^
  packaging\launcher.py || goto :error
echo.
echo 打包完成：dist\MiniscopeAnalysis\MiniscopeAnalysis.exe
goto :eof
:error
echo 打包失败，请把上面的报错信息发给维护者。
exit /b 1
