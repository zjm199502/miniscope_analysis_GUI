param([Parameter(Mandatory)][string]$Version)
$ErrorActionPreference = "Stop"
$compiler = @(
    (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (!$compiler) { throw "Install Inno Setup 6 to build CalciLume-Windows-Setup.exe" }
& $compiler "/DAppVersion=$Version" "$PSScriptRoot\calcilume.iss"
exit $LASTEXITCODE
