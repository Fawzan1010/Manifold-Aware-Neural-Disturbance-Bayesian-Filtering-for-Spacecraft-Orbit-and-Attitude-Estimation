# Runs the three-storm extension (17 Mar, 22 Jun, 20 Dec 2015) sequentially,
# reusing the already-trained PINN/Transformer checkpoints in outputs/models.
# Each run's storm/ output is archived before the next run starts, since
# main.py always writes to <output_dir>/storm.
#
# Expect this to take on the order of a day of wall-clock time. Run it in a
# window that will not sleep/close (or via Task Scheduler), e.g.:
#   powershell -ExecutionPolicy Bypass -File run_storm_all.ps1
#
# When all three finish, rebuild the corrected Table 8 with:
#   venv\Scripts\python.exe build_table8.py outputs\storm_17mar:17Mar outputs\storm_22jun:22Jun outputs\storm_20dec:20Dec

$python = ".\venv\Scripts\python.exe"
$ErrorActionPreference = "Stop"

function Run-Storm($config, $archiveName) {
    Write-Host "=== Running $config ===" -ForegroundColor Cyan
    & $python main.py --config $config --mode storm
    if ($LASTEXITCODE -ne 0) { throw "Storm run failed for $config" }
    if (Test-Path "outputs\$archiveName") { Remove-Item -Recurse -Force "outputs\$archiveName" }
    Move-Item "outputs\storm" "outputs\$archiveName"
    Write-Host "=== Archived to outputs\$archiveName ===" -ForegroundColor Green
}

Run-Storm "configs\storm_17mar.yaml" "storm_17mar"
Run-Storm "configs\storm_22jun.yaml" "storm_22jun"
Run-Storm "configs\storm_20dec.yaml" "storm_20dec"

Write-Host "All three storm runs complete. Run build_table8.py next." -ForegroundColor Yellow
