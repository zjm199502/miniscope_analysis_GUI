param([Parameter(Mandatory)][string]$Version)
$ErrorActionPreference = "Stop"
$setup = (Resolve-Path "dist/CalciLume-Windows-Setup.exe").Path
$scratch = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$area = Join-Path $scratch ("calcilume-install-test-" + [guid]::NewGuid())
$install = Join-Path $area "app"
New-Item -ItemType Directory -Path $area | Out-Null
$project = Join-Path $area "project.json"
Set-Content -LiteralPath $project -Value "external project"
function Run-Setup($program, $arguments) {
    $p = Start-Process -FilePath $program -ArgumentList $arguments -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "$program returned $($p.ExitCode)" }
}
# Exercise the same installation twice to cover in-place replacement and the uninstall entry.
foreach ($pass in 1..2) {
    Run-Setup $setup @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/DIR=`"$install`"")
    $exe = Join-Path $install "runtime/MiniscopeAnalysis.exe"
    $link = Join-Path ([Environment]::GetFolderPath("Programs")) "CalciLume.lnk"
    if (!(Test-Path -LiteralPath $link)) { throw "Missing start menu shortcut: $link" }
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
    $target = $shortcut.TargetPath
    python -c "import os,sys; assert os.path.samefile(sys.argv[1], sys.argv[2]), sys.argv[1:]" "$target" "$exe"
    if ($LASTEXITCODE) { throw "Shortcut target '$target', expected '$exe'" }
    $p = Start-Process -FilePath $target -WorkingDirectory $install -PassThru
    try {
        $deadline = (Get-Date).AddSeconds(60)
        do { Start-Sleep -Milliseconds 300; $p.Refresh() } until ($p.HasExited -or $p.MainWindowTitle -or (Get-Date) -gt $deadline)
        if ($p.MainWindowTitle -notlike "CalciLume*$Version*") { throw "Unexpected window: $($p.MainWindowTitle)" }
        $null = $p.CloseMainWindow()
        if (!$p.WaitForExit(10000)) { throw "Window close did not release process" }
    } finally { if (!$p.HasExited) { $p.Kill() } }
}
Run-Setup (Join-Path $install "unins000.exe") @("/VERYSILENT", "/NORESTART")
if ((Test-Path $exe) -or (Test-Path $link) -or !(Test-Path $project)) { throw "Uninstall check failed" }
