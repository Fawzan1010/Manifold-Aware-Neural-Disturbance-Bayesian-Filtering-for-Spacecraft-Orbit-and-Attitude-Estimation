# Changelog

Notes on the evolution of the codebase, kept for provenance.  In-code
revision tags have been removed; this file is the record of what changed.

---

## 0. Ground-truth leakage in the estimator loop

**Where:** `src/evaluation/experiments.py` — v1 `_run_filter` (PINN+UKF+MEKF
branch) and `_run_hybrid_ukf_mekf`.

**Problem:** v1 corrected the estimated quaternion toward the *true* simulator
state every step:

```python
q_ref = traj.states[k, :4]          # simulator TRUTH
dq = quat_multiply(q_ref, quat_conjugate(q_est))
qerr = rotvec_from_quat(dq)
dq_small = np.hstack([1.0, 0.5 * gain * qerr])   # gain 0.5–0.7
filt.x[:4] = normalize_quaternion(quat_multiply(dq_small, filt.x[:4]))
```

No estimator may access `traj.states` at runtime. This inflated the 47.5%
attitude improvement and would be grounds for rejection if a reviewer ran the
code.

**Fix (v2):** removed entirely. `_run_hybrid_ukf_mekf` now implements the
architecture the paper actually describes: a UKF for the translational
partition and a real MEKF for the rotational partition, both updated only from
noisy measurements, cross-feeding partitions and fused per Section IX.

**Consequence:** all headline numbers (47.5% attitude, 4.0% position, 13.1 ms)
must be regenerated with `python main.py --mode all` and the paper's abstract,
Section XIII, and conclusions updated to the new values.

---

## 1. "No proof of convergence / stability / bounded error / observability"

**New file:** `src/evaluation/theory.py` — run `python main.py --mode theory`
(also runs in `--mode all`). Produces `outputs/theory_report.json` with:

- **Observability:** rank, condition number, and minimum singular value of the
  local discrete observability matrix `O = [H; HF; HF²; …]` built from
  finite-difference Jacobians along test trajectories → supports an empirical
  local-observability claim for the 25-dim state.
- **Consistency:** fraction of NEES/NIS samples inside 95% chi-square bounds.
- **Bounded error:** per-method fit of `‖e_k‖ ≤ a·ρᵏ + b` (ρ < 1 with small b
  = numerical evidence of exponential convergence to a bounded residual set).
- **Lyapunov candidate:** drift statistics of `V_k = e_kᵀP_k⁻¹e_k`
  (negative expected drift above the chi-square bound supports stochastic
  boundedness).

**Paper side:** add a "Stability and Observability Analysis" section: state a
bounded-error proposition for the MEKF/UKF error dynamics under standard
assumptions (uniform observability, bounded Jacobians, PINN prediction error
bounded — an input-to-state stability argument in the style of Reif et al.,
"Stochastic stability of the discrete-time extended Kalman filter"), and cite
`theory_report.json` numbers as the empirical verification.

**Related bug fixed:** v1 NEES was always NaN (24-dim error vector
vs 25-dim covariance). `_error_covariance()` in `experiments.py` now maps the
quaternion covariance to the 24-dim error-state space (δθ ≈ 2δq_v), so NEES
plots/tables are real.

---

## 2. "Why is your algorithm 9× faster than UKF? No justification"

It wasn't — it was a measurement artifact. Two bugs:

- **Where:** `src/evaluation/experiments.py` — PINN inference
  (`_predict_residual`) ran *outside* `profile_block`, so hybrid methods were
  never charged for NN inference.
- **Where:** `src/utils/profiling.py` usage — `psutil.memory_info()` was called
  twice per step *inside* the timed region; its variable overhead dominated
  and distorted per-method comparisons (same UKF step: 122.7 ms standalone vs
  36.8 ms in PINN+UKF).

**Fix (v2):** `time.perf_counter` around the **full estimation cycle**
(PINN inference + injection + filter step(s)); memory sampled once per
trajectory; new per-component breakdown columns `pinn_time_per_step_s`,
`filter_time_per_step_s` and new tables
`outputs/tables/runtime_breakdown.{csv,tex}` (PINN ms + filter ms = total ms,
with std and median). Note the honest v2 runtime of PINN+UKF+MEKF will be
*larger* than UKF alone (it runs a UKF **and** an MEKF per step) — update
Section XIII-F and Table (runtime) accordingly and remove the
"lowest runtime" claim.

---

## 3. "100 Monte Carlo runs but no confidence intervals, box plots, variance"

**Where:** `src/evaluation/stats.py`, `src/evaluation/experiments.py`,
`src/visualization/plots.py`.

- `stats.py`: added `bootstrap_ci`, `wilcoxon_test`, `mc_summary`
  (mean/median/std/var/t-CI/bootstrap-CI/IQR).
- `experiments.py` now writes `outputs/tables/metrics_per_trial.csv` (100 rows
  × method) and adds to `metrics_summary.csv`: `*_std`, `*_var`,
  `*_ci95_lo/hi`, plus paired t-test **and** Wilcoxon signed-rank p-values vs
  EKF for position and attitude.
- `plots.py`: box plots over the 100 trials for position, attitude, velocity,
  angular-rate, runtime (`*_boxplot.pdf`) and mean ± 95% CI bar charts
  (`*_ci.pdf`). v1's "boxplots" were drawn from the one-row-per-method summary
  (a single point per box). Empirical CDFs now also use per-trial data.

**Paper side:** report every headline number as `mean ± 95% CI (n = 100)`,
include the attitude & position boxplots, and quote Wilcoxon p-values
alongside t-tests (RMSE distributions are not Gaussian).

---

## 4. "No discussion of PINN depth, window length, loss weights, learning rate, coupling coefficient"

**New file:** `src/evaluation/ablation.py` — run `python main.py --mode ablate`.
One-at-a-time sweep (3 seeds each) over: `depth` {2,4,6}, `hidden`
{64,128,256}, `window` {2,4,8}, `lr` {3e-4,1e-3,3e-3}, `lambda_norm`
{0,0.1,1.0}, `coupling` {0.25,0.5,0.75,0.98}. Outputs
`outputs/tables/ablation_results.csv`, `ablation_summary.csv`, and
`outputs/figures/ablation_<axis>.pdf` (mean ± std error bars).

**Also fixed:**
- `src/pinn/model.py`: depth/activation are now arguments.
  v1 hard-coded **3 hidden layers with SiLU**, while the paper claims **H=4,
  tanh** — v2 defaults now match the paper. Either keep the v2 defaults or
  correct the paper text.
- `configs/default.yaml`: coupling coefficients moved from
  hard-coded values in `_inject_residual` to `fusion.coupling`.

**Paper side:** add a "Hyperparameter Sensitivity" subsection with the
ablation figures and a table of chosen values + sweep ranges.

---

## 5. "Show #epochs, training time, GPU, batch size, window size, dataset split, noise values"

**Where:** `src/pinn/train.py`, `src/models/train.py`.

Training now writes `outputs/models/pinn_training_log.json` and
`transformer_training_log.json` containing: epochs run (with early stopping,
patience 10), best epoch, wall-clock training time, device + GPU/CPU name,
parameter count, batch size, window size, learning rate + schedule
(cosine annealing, matching Algorithm 1 in the paper — v1 had *no* scheduler),
loss weights, optimizer, gradient-clip norm, train/val sample and trajectory
counts, and full per-epoch train/val loss curves.

Dataset split (80 train / 20 val / 100 test) and all noise values were already
in `configs/default.yaml` (echoed to `outputs/metadata.json`); cite that file.

**Paper side:** add a "Training and Implementation Details" table populated
directly from the two JSON logs.

---

## 6. "The paper does not use [a standard] simulator"

Code already ships a self-contained simulator (`src/dynamics/simulator.py`);
what is missing is *independent validation*. Recommended (not automated here):

1. Cross-validate the propagator against an established tool (e.g., Orekit or
   GMAT) on one orbit: same initial conditions, compare position error over
   240 s; report the deviation in an appendix.
2. State explicitly in Section XI that the benchmark is synthetic, and cite
   the planned GRACE-FO/Swarm loaders (`src/sensors/loaders.py`) as future
   validation on flight data.

---

## How to regenerate everything

```bash
pip install -r requirements.txt
python main.py --mode all       # all seven stages, in order
python main.py --mode ablate    # hyperparameter sensitivity only (slow)
```

New outputs: `tables/metrics_per_trial.csv`, `tables/runtime_breakdown.csv`,
`tables/ablation_*.csv`, `models/*_training_log.json`, `theory_report.json`,
`figures/*_boxplot.pdf`, `figures/*_ci.pdf`, `figures/ablation_*.pdf`.

---

# v2.1 addendum — performance upgrades (legitimate accuracy improvements)



1. **Bayesian pseudo-measurement fusion** (`experiments.py::_fuse_pinn_prior`).
   The PINN prior is no longer convex-blended into the state. It is fused as a
   pseudo-measurement on the disturbance sub-states with
   `R_pinn = diag(PINN predicted variance) * fusion.pinn_r_scale`, via a full
   Kalman update. Three real benefits: (a) the PINN's heteroscedastic
   uncertainty head (logvar) is finally used — confident predictions pull
   hard, uncertain ones barely act; (b) cross-covariances propagate the
   disturbance correction into position/velocity/attitude; (c) the covariance
   contracts, so NEES/NIS consistency improves, not just RMSE.
2. **Training budget** (`configs/default.yaml`): pinn_epochs 50→150 with early
   stopping (patience 15), window 4→8 (longer disturbance history), cosine LR
   annealing (added in v2), transformer_epochs 50→80.
3. **Ablation axis** `coupling` replaced by `pinn_r_scale` {0.25, 1, 4, 16};
   tune this first — it is the single most impactful knob.

Expected effect: gains concentrate where the paper's story is —
high-disturbance / storm / maneuver scenarios, where fixed-Q Gauss-Markov
filters are mismatched. Report scenario-stratified tables to show this.
No number in the paper should be written down before `--mode all` completes.

---

## Later revision

Independent of the items above:

- Atmospheric density is now a pluggable model (`src/dynamics/atmosphere.py`):
  the exponential profile is retained as the default and reproduces previous
  results bit-for-bit, with NRLMSISE-00 available via `pymsis` driven by
  measured OMNI2 F10.7/Ap.
- Fixed a unit error in `drag_acceleration`: the reference area is in m^2 and
  speed in km/s, which left the result short by a factor of 1e3.
- NEES is reported for the full 24-dim error state and separately for the
  12-dim rotational and translational partitions.
- Runtime is measured in a dedicated pass with methods interleaved inside the
  repeat loop and the leading repeat discarded.
- Added `--mode storm`: evaluation of every estimator over the 17 March 2015
  geomagnetic storm, reusing the trained checkpoints without retraining.

---

## Star-tracker attitude sensing (this revision)

**Where:** `src/dynamics/spacecraft.py` (`SpacecraftParams.star_tracker`,
`Environment.att_ref1_i/att_ref2_i`, `environmental_fields`),
`src/sensors/measurement_models.py` (`measurement_vector`,
`measurement_noise_cov`), `src/dynamics/simulator.py`,
`src/evaluation/experiments.py` (`_env_at`, the two `R` construction sites).

**Problem:** every reported attitude number in the paper (benchmark and
storm) was conditioned on a two-vector geometry (sun direction and nadir)
that closes to ~171 deg apart over an equatorial orbit, amplifying
measurement sensitivity by a factor of ~6.7 (docs/sanity_check). No star
tracker was modelled; "arcsecond-level" language in the paper was not
supported by the evaluated sensor suite.

**Fix:** `simulation.star_tracker: true` (now default in
`configs/default.yaml` and all three storm configs) switches
`environmental_fields()` to populate two fixed, near-orthogonal
(~90 deg apart) catalog-star reference directions, decoupled from the
physical sun vector (which is retained unchanged for SRP). The measurement
model consumes these through new `Environment.att_ref1_i/att_ref2_i` fields
(falling back to the legacy sun/nadir pair when unset, so `star_tracker:
false` reproduces the original geometry and noise exactly).
`measurement_noise_cov` gains a `star_tracker` flag that selects
`measurement_noise.vector_st` (20 arcsec/axis, matching
docs/sanity_check/star_tracker_diag.py) in place of the old
`measurement_noise.vector` (0.012 rad, ~41 arcmin) when enabled.

**Verified:** a regression check driving `environmental_fields` +
`measurement_vector` + `measurement_noise_cov` + `MEKF.step` through the
actual (not standalone) pipeline gives steady-state median attitude error of
~26 arcsec with `star_tracker: true` and ~5900 arcsec with `star_tracker:
false` (bit-reproducing the pre-change geometry) -- confirming the fix is
real and the legacy path is unaffected. The existing test suite
(`tests/test_quaternion.py`, `test_filters.py`, `test_frames.py`,
`test_atmosphere.py`, `test_space_weather.py`) passes unchanged.

**Consequence:** every number in the paper (Tables 4/6/7/8/9, all synthetic
and storm figures) must be regenerated with a star tracker in the loop --
`run_everything.ps1` does the full 100-trajectory benchmark, training, and
all three storm arcs (17 Mar / 22 Jun / 20 Dec, 8 arcs total) end to end.
