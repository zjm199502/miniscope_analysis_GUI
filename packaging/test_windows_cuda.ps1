param(
    [string]$Exe = "dist/MiniscopeAnalysis/MiniscopeAnalysis.exe",
    [string]$Report,
    [switch]$RequireGpu
)
$ErrorActionPreference = "Stop"
$program = (Resolve-Path -LiteralPath $Exe).Path
$scratch = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$area = Join-Path $scratch ("calcilume-cuda-check-" + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $area
if (!$Report) { $Report = Join-Path $area "runtime.json" }
$Report = [IO.Path]::GetFullPath($Report)
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $program
$info.WorkingDirectory = $area
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.Arguments = '--check-windows-cuda "' + $Report + '"'
if ($RequireGpu) { $info.Arguments += ' --require-gpu' }
$info.EnvironmentVariables["PATH"] = "$env:SystemRoot\System32;$env:SystemRoot"
$info.EnvironmentVariables["CUPY_CACHE_DIR"] = $area
foreach ($name in @($info.EnvironmentVariables.Keys)) {
    if ($name -match '^(CUDA_PATH.*|CUDA_HOME|CONDA_PREFIX|VIRTUAL_ENV|PYTHONHOME|PYTHONPATH)$') {
        $info.EnvironmentVariables.Remove($name)
    }
}
$process = [Diagnostics.Process]::Start($info)
if (!$process.WaitForExit(180000)) { $process.Kill(); throw "Packaged CUDA self-test timed out" }
if (!(Test-Path -LiteralPath $Report)) { throw "No CUDA report (exit $($process.ExitCode))" }
$json = Get-Content -Encoding UTF8 -Raw -LiteralPath $Report
Write-Output $json
if ($process.ExitCode -ne 0 -or !($json | ConvertFrom-Json).ok) {
    throw "Packaged CUDA self-test failed (exit $($process.ExitCode))"
}
