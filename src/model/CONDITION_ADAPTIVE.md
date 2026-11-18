# Condition-adaptive RBD/KAN v1

This offline A0509 study uses the existing combined legacy + wide16_v3 archive at
`F:\robot_modeling\result\combined_legacy_wide_v3\prepared`. Its target is
`tau_reference` (saved MuJoCo inverse dynamics), with `tau_nominal` as the
nominal rigid-body baseline. The separate J1 friction-compensation pilot and
`tau_friction` are outside this study.

## Run

The current result folder is
`F:\robot_modeling\result\condition_adaptive_rbd_kan_v1`. A complete fresh
run needs a new output path:

```powershell
Set-Location F:\robot_modeling
.\scripts\run_condition_adaptive.ps1 -Stage All -OutputRoot F:\robot_modeling\result\condition_adaptive_rbd_kan_v2
```

For a partially completed run, use the same output path with `-Stage Audit`,
`-Stage Basis`, `-Stage Train`, or `-Stage Figures` in order. `Train` reuses
existing checkpoints only when the configuration and split sizes match. The
script uses the repository's `.venv` by default and reads saved MuJoCo scenes
and RBD code from `D:\Code\sdml-doosan-robot`.

## Audit and split

The code checks source and prepared SHA-256 digests, 296,520 rows, 22 features,
seven conditions, all 16 trajectories, finite values, original and nominal
scene-label agreement, and full-row inertial-scale consistency. The AG145 XMLs
have different mesh path strings across campaigns; their other content and
mesh bytes agree.

Train uses T1–T11/T15, validation T12, and test T13/T14/T16. An exact-state
audit found 28 unique feature rows shared by train and validation, appearing
182 times in validation. Those validation rows are excluded before model
selection. There are zero exact train/test and validation/test feature overlaps.

For the 0.5 kg holdout, both AG145 and mDPG 0.5 kg conditions are absent from
train and validation. The 1.0 kg holdout follows the same rule. Bare and the
other two loads for each gripper remain available. All final tests use
T13/T14/T16.

## Models

- **R0:** saved nominal RBD prediction (0.8 attachment-inertia scale).
- **R1:** one shared physical inertia-scale increment, fitted by weighted
  least squares on training labels. The physical sensitivity is computed from
  MuJoCo nominal scenes at scales 0.6 and 0.8 without reference labels.
- **R2:** R0 plus six independent velocity splines (54 trainable parameters).
- **R3:** R0 plus a 22→30→30→6 MLP (1,806 parameters).
- **R4:** R0 plus a 22→8→6 edge-spline KAN (1,806 parameters).

R3 and R4 use identical 22 inputs, three seeds (11/29/47), an equal 30-epoch
ceiling, Adam learning rate 0.001, batch size 2,048, identical validation
policy, and equal total loss weight for each campaign. R2 has the same training
budget but deliberately restricted input. All models evaluate the same test
rows within each protocol. Test results are not used to select settings.

## Outputs and interpretation

`REPORT.md` gives the completed numerical results and separates supported
claims from limitations. `inertia_residual_7x6.csv` contains RMSE, MAE,
maximum absolute error and 95th percentile absolute error for each condition
and joint. `model_metrics.csv` contains test, condition, joint and campaign
metrics per seed, with parameter counts and training time. `figures/`
contains PNG, SVG and source CSV for Figures 1–6.

The original 0.8-scale mismatch is a known synthetic perturbation of the saved
scene. R1 can therefore almost exactly reverse it; that result cannot be
interpreted as identification of unknown real-robot parameters. No result in
this study is measured motor torque or sim-to-real performance.

## Completed v1 result

The full local reports are
`F:\robot_modeling\result\condition_adaptive_rbd_kan_v1\REPORT_KO.md` and
`REPORT.md`. On the pooled T13/T14/T16 test, torque RMSE was:

| Model | RMSE (Nm) | Mean training time |
| --- | ---: | ---: |
| R0 nominal RBD | 0.676604 | 0 s |
| R1 one-scale physical correction | 0.00000145 | 0.016 s scalar fit* |
| R2 joint-wise velocity spline | 0.682534 ± 0.000235 | 4.90 s |
| R3 22-input MLP | 0.050404 ± 0.0106 | 7.14 s |
| R4 22-input KAN | 0.134514 ± 0.0101 | 88.71 s |

The `±` value is the sample standard deviation over three seeds.
The R1 scalar-fit time excludes physical-sensitivity preprocessing.
R1 nearly exactly recovers the known synthetic 0.8-to-1.0 scale change.
In the fixed-budget R3/R4 comparison, MLP had lower error and training time.
Both models selected the 30-epoch cap in all pooled seeds, so this does not
establish their fully converged ranking.
For R3, entirely withholding 0.5 kg increased same-condition test RMSE by
1.08× (AG145) and 1.09× (mDPG); withholding 1.0 kg increased it by 1.65×
and 1.35×. These results do not establish real-robot torque accuracy or a
general ranking of neural architectures.
