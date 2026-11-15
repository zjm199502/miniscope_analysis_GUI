@echo off
chcp 65001 >nul
REM 不打包、直接运行：第一次会自动建立运行环境（约 5 分钟），之后双击即可启动
cd /d %~dp0\..
if not exist .venv\Scripts\python.exe (
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -e .[fast]
)
start "" .venv\Scripts\pythonw.exe -m miniscope_gui
