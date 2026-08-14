# PILOT Paper Artifact

This repository contains the reproducibility artifact for the PDSW paper introducing PILOT: Online Compression Auto-Tuning for Streaming Light-Source Imaging. PILOT provides online prediction and selection of error-bounded lossy-compression configurations for streaming light-source data.

The artifact contains the complete workflow for reproducing the paper’s quantitative results. It uses five archived CSV files containing compression experiment results to train compressor-specific Ridge models, evaluate online configuration selection, validate deterministic results, and regenerate the paper’s tables and figures. The main workflow does not rerun the lossy-compression experiments. However, the compression experiment scripts are included for reference and can be executed separately as needed.

Repository: <https://github.com/JiaxiangCxx/PILOT_PDSW26>

## Obtain the artifact

```bash
git clone https://github.com/JiaxiangCxx/PILOT_PDSW26.git
cd PILOT_PDSW26
git lfs pull
```

Git LFS is required for the large SFC-L input CSV.

## Included files

- `artifact/environment.yml`: verified CPU-only Conda environment.
- `artifact/setup_env.sh`: creates an isolated environment inside this repository.
- `artifact/paper_experiment_manifest.csv`: authoritative list of the 21 paper runs.
- `artifact/check_artifact.py`: validates inputs, schemas, checksums, and references.
- `artifact/compare_reference.py`: compares deterministic run-summary metrics.
- `artifact/run_all_paper_experiments.sh`: sequential runner for all paper experiments.
- `lightsource_data_cmp_results/*/*compression.py`: original optional
  compression-experiment scripts.
- `.gitignore`: excludes regenerated outputs and local caches.

The runner additionally uses:

- `model/multi_metric_joint_model.py`
- `model/reproduce_paper_results.ipynb`
- the five CSVs listed in `artifact/paper_experiment_manifest.csv`
- the committed reference summaries listed in the same manifest

## Optional Compression Experiments

The original compression experiment scripts are included unchanged for
reference and reviewer convenience. They are not invoked by the default
artifact workflow because the model experiments use the archived compression-result CSV files included in this repository.

Reviewers may run the compression scripts separately as needed. Their
execution requires the original light-source datasets, LibPressio, and the
corresponding compressor plugins. The required datasets are available through
the following link:

- [PILOT light-source datasets](https://anl.box.com/s/5bdz7y6j9bt9dz6pkqjhvech70kfude8)

Because these compression experiments depend on external software and large
datasets, they are optional and are not required to reproduce the paper's
model-level tables and figures.

## Requirements

- A Linux system is required. The artifact was validated on an x86-64 CPU; no GPU is required.
- At least 16 GB RAM and sufficient space for the input and generated CSVs.
- Python 3.10 and the packages pinned in `artifact/environment.yml`.

Create a project-local environment from the repository root:

```bash
bash artifact/setup_env.sh
conda activate "$PWD/.conda/pilot-ae"
```

The `.conda/` directory is ignored by Git. After activation, the runner resolves
`python` and `jupyter` from this environment rather than using a user-global
environment or a host-specific absolute path.

## Validate the artifact

From the repository root:

```bash
python artifact/check_artifact.py
bash artifact/run_all_paper_experiments.sh --list
```

The first command must print `Artifact validation passed.` The second command
must list exactly 21 experiments.

## Run all paper experiments

```bash
bash artifact/run_all_paper_experiments.sh
```

The runner executes experiments sequentially so that the end-to-end wall time
is interpretable. Each newly generated result is checked against its committed
reference before the next experiment starts. A mismatch or missing file stops
the run immediately.

The final terminal output must contain:

```text
All 21 paper experiments completed and matched their references.
```

## Paper Experiment Manifest

[`artifact/paper_experiment_manifest.csv`](artifact/paper_experiment_manifest.csv) is the authoritative list of the 21 model experiments used in the paper. Each row specifies the dataset, optimization objective, training and prediction window sizes, tuning granularity, input compression-result CSV, expected output location, and corresponding paper table or figure. The artifact workflow reads this manifest to execute and validate the paper experiments consistently.

## Generated outputs

Every execution replaces the fixed output tree:

```text
artifact/output/
  datasets/<dataset>/<objective>/W<W>_P<P>_tau<tau>/
    compressor_timing.csv
    joint_prediction_accuracy.csv
    joint_selection_comparison.csv
    run_summary.log
  ad_execution_time.txt
```

Expected completeness:

- 21 experiment directories and 84 model-result files;
- one complete sequential wall-time record for the AD appendix.

The notebook renders the paper figures and tables during execution but does not
save duplicate copies under `artifact/output/`.

The notebook is a post-processing step, so it cannot run in a fresh clone before
the batch has created `artifact/output/datasets/`. The full runner executes the
notebook automatically after all 21 experiments pass validation. To inspect a
previously generated output tree without rerunning the experiments, launch
Jupyter with `PILOT_OUTPUT_ROOT=/path/to/artifact/output`.

`artifact/output/` is intentionally ignored by Git. It is regenerated during
evaluation and must not be committed as a second copy of the reference data.

## Paper mapping

- Table III: five dataset-wide CR-PSNR-SSIM runs at `W/P=16/16`, `tau=1`.
- Table IV: SFC-L CR, PSNR, SSIM, CR-PSNR, and CR-PSNR-SSIM objectives.
- Table V: five dataset-wide CR-PSNR runs.
- Table VI: SFC-L window and tuning-granularity sensitivity.
- Figure 5: FF-HEDM actual-versus-predicted metrics.
- Figure 6: FF-HEDM frame-32 rate-quality curves and PILOT selections under
  PSNR, CR-PSNR, and CR-PSNR-SSIM objectives.

Figures 1 and 2 are regenerated directly from the archived input CSVs by the
paper notebook. Figures 3 and 4 are outside this model-level workflow because they either
require application arrays or do not introduce additional numerical
experiments.

## Runtime interpretation

`artifact/output/ad_execution_time.txt` reports the complete artifact wall
time, including the 21 model runs, reference validation, and notebook
post-processing. It is distinct from the online time and speedup reported by
the model in each `run_summary.log`.
