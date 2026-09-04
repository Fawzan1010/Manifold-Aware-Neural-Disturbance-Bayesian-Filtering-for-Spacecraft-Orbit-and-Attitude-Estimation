# ============================================================================
# Full Q1-submission run: star-tracker sensing, full 100-trajectory synthetic
# benchmark (training + evaluation + theory + plots + ablation), plus the
# complete 3-storm transfer study (17 Mar / 22 Jun / 20 Dec 2015, 8 arcs
# total), all reusing the same trained checkpoints as required by the paper's
# "nothing retrained or retuned" claim.
#
# Star tracker: configs/default.yaml and all three storm configs now set
# simulation.star_tracker = true. This switches the attitude-measurement
# reference pair from the ~171-deg sun/nadir geometry (ill-conditioned by a
# factor of ~6.7, per docs/sanity_check) to a fixed, near-orthogonal
# catalog-star pair at sigma = 20 arcsec/axis -- confirmed by
# docs/sanity_check/star_tracker_diag.py and by the regression check in this
# revision to converge the MEKF to ~26 arcsec steady-state median under
# disturbance-free conditions (vs ~5900+ arcsec under the old two-vector
# geometry). Everything downstream (training data, benchmark, storm arcs)
# now uses the star-tracker measurement by default.
#
# Expect this to take on the order of a day of wall-clock time (the 100-
# trajectory benchmark + training dominate; each storm arc is comparatively
# short). Run it in a window that will not sleep, or via Task Scheduler:
#   powershell -ExecutionPolicy Bypass -File run_everything.ps1
# ============================================================================

$python = ".\venv\Scripts\python.exe"
$ErrorActionPreference = "Stop"

function Run-Stage($mode) {
    Write-Host "=== main.py --mode $mode (star tracker) ===" -ForegroundColor Cyan
    & $python main.py --config configs\default.yaml --mode $mode
    if ($LASTEXITCODE -ne 0) { throw "Stage '$mode' failed" }
}

function Run-Storm($config, $archiveName) {
    Write-Host "=== Storm: $config ===" -ForegroundColor Cyan
    & $python main.py --config $config --mode storm
    if ($LASTEXITCODE -ne 0) { throw "Storm run failed for $config" }
    if (Test-Path "outputs\$archiveName") { Remove-Item -Recurse -Force "outputs\$archiveName" }
    Move-Item "outputs\storm" "outputs\$archiveName"
    Write-Host "=== Archived to outputs\$archiveName ===" -ForegroundColor Green
}

Write-Host "--- 0. Star-tracker sanity check (fast, ~1-2 min) ---" -ForegroundColor Yellow
& $python docs\sanity_check\star_tracker_diag.py --horizon 2000 --rscale 2.0 --qscale 0.5 --tag final_check --out docs\sanity_check\r_final.json
if ($LASTEXITCODE -ne 0) { throw "Star-tracker sanity check failed -- stop and inspect before running the full benchmark." }
Write-Host "Sanity check passed. See docs\sanity_check\r_final.json for the steady-state arcsecond number to cite in the paper." -ForegroundColor Green

Write-Host "--- 1. Full synthetic benchmark (synth / train / evaluate / theory / plot / ablate) ---" -ForegroundColor Yellow
Run-Stage "synth"
Run-Stage "train"
Run-Stage "evaluate"
Run-Stage "theory"
Run-Stage "plot"
Run-Stage "ablate"

Write-Host "--- 2. Three-storm transfer study (8 arcs: 17 Mar x4, 22 Jun x2, 20 Dec x2) ---" -ForegroundColor Yellow
Run-Storm "configs\storm_17mar.yaml" "storm_17mar"
Run-Storm "configs\storm_22jun.yaml" "storm_22jun"
Run-Storm "configs\storm_20dec.yaml" "storm_20dec"

Write-Host "--- 3. Rebuilding the corrected, common-survivor-restricted Table 8 ---" -ForegroundColor Yellow
& $python build_table8.py "outputs\storm_17mar:17Mar" "outputs\storm_22jun:22Jun" "outputs\storm_20dec:20Dec" | Tee-Object -FilePath "outputs\table8_final.txt"

Write-Host ""
Write-Host "ALL DONE." -ForegroundColor Green

