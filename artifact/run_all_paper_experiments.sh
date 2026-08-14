#!/usr/bin/env bash
set -euo pipefail

# PILOT paper experiment coverage (21 unique model runs; no compression reruns):
#   1-5   CSSI, FF-HEDM, SFC-GI, SFC-L, XPCS: CR-PSNR-SSIM, W/P=16/16, tau=1
#   6-9   SFC-L objective study: CR, PSNR, SSIM, CR-PSNR, W/P=16/16, tau=1
#   10-12 SFC-L window study: 8/8, 32/32, 32/512, tau=1
#   13-16 SFC-L granularity study: 32/512, tau=4,16,64,256
#   17-20 CSSI, FF-HEDM, SFC-GI, XPCS: CR-PSNR, W/P=16/16, tau=1
#   21    FF-HEDM: PSNR, W/P=16/16, tau=1
# Paper mapping: Table III uses runs 1-5; Table IV uses runs 4 and 6-9;
# Table V uses runs 9 and 17-20; Table VI uses runs 10-16;
# Figure 5 uses run 2; Figure 6 uses runs 2, 18, and 21.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=$ROOT/artifact/paper_experiment_manifest.csv
MODEL=$ROOT/model/multi_metric_joint_model.py
COMPARE=$ROOT/artifact/compare_reference.py
NOTEBOOK=$ROOT/model/reproduce_paper_results.ipynb
PYTHON=${PYTHON:-python}
JUPYTER=${JUPYTER:-jupyter}
OUTPUT_ROOT=$ROOT/artifact/output
DATASET_ROOT=$OUTPUT_ROOT/datasets
TIMING_FILE=$OUTPUT_ROOT/ad_execution_time.txt
LIST_ONLY=0
RUN_NOTEBOOK=1

usage(){
  cat <<EOF
Usage: bash artifact/run_all_paper_experiments.sh [options]

Options:
  --list          Print all 21 experiments without running them.
  --no-notebook   Skip execution of reproduce_paper_results.ipynb.
  -h, --help      Show this help.

The fixed output is artifact/output/. A rerun replaces datasets/ and
ad_execution_time.txt. The script reads precomputed CSVs and never runs a
compressor.
EOF
}

while (($#));do
  case "$1" in
    --list) LIST_ONLY=1;shift;;
    --no-notebook) RUN_NOTEBOOK=0;shift;;
    -h|--help) usage;exit 0;;
    *) echo "Unknown option: $1" >&2;usage >&2;exit 2;;
  esac
done

list_experiments(){
  printf "%-3s %-9s %-14s %-8s %-6s %s\n" "#" "Dataset" "Objective" "W/P" "tau" "Paper elements"
  tail -n +2 "$MANIFEST" | while IFS=',' read -r order dataset objective w p tau max_cr csv_path reference paper;do
    printf "%-3s %-9s %-14s %-8s %-6s %s\n" "$order" "$dataset" "$objective" "$w/$p" "$tau" "$paper"
  done
}

[[ -f $MANIFEST ]]||{ echo "Missing manifest: $MANIFEST" >&2;exit 1;}
if ((LIST_ONLY));then list_experiments;exit 0;fi
PYTHON_REQUESTED=$PYTHON
PYTHON=$(command -v "$PYTHON_REQUESTED")||{ echo "Python was not found: $PYTHON_REQUESTED" >&2;exit 1;}
[[ -f $MODEL ]]||{ echo "Missing model: $MODEL" >&2;exit 1;}
[[ -f $COMPARE ]]||{ echo "Missing comparison script: $COMPARE" >&2;exit 1;}
grep -q "decision feature extraction time" "$MODEL"||{
  echo "Model does not contain corrected tau-aware feature timing: $MODEL" >&2;exit 1;
}

# Validate all inputs before deleting a previous fixed-output run.
tail -n +2 "$MANIFEST" | while IFS=',' read -r order dataset objective w p tau max_cr csv_path reference paper;do
  [[ -f $ROOT/$csv_path ]]||{ echo "Missing input CSV: $ROOT/$csv_path" >&2;exit 1;}
  [[ -f $ROOT/$reference/run_summary.log ]]||{ echo "Missing reference: $ROOT/$reference" >&2;exit 1;}
done
if ((RUN_NOTEBOOK));then
  JUPYTER_REQUESTED=$JUPYTER
  JUPYTER=$(command -v "$JUPYTER_REQUESTED")||{ echo "Jupyter was not found: $JUPYTER_REQUESTED" >&2;exit 1;}
  [[ -f $NOTEBOOK ]]||{ echo "Missing notebook: $NOTEBOOK" >&2;exit 1;}
fi

[[ $OUTPUT_ROOT == "$ROOT/artifact/output" ]]||{ echo "Unsafe output root: $OUTPUT_ROOT" >&2;exit 1;}
rm -rf -- "$DATASET_ROOT"
rm -f -- "$TIMING_FILE"
mkdir -p "$DATASET_ROOT"

TOTAL=$(($(wc -l < "$MANIFEST")-1))
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OVERALL_START_NS=$(date +%s%N)

echo "PILOT paper reproduction"
echo "Output: $OUTPUT_ROOT"
echo "Input: precomputed compression CSVs only"
echo "Execution: sequential"
echo
list_experiments
echo

while IFS=',' read -r order dataset objective w p tau max_cr csv_path reference paper;do
  [[ $order == order ]]&&continue
  result=$DATASET_ROOT/$dataset/$objective/W${w}_P${p}_tau${tau}
  command=(
    "$PYTHON" "$MODEL"
    --csv-path "$ROOT/$csv_path"
    --objective "$objective"
    --train-window "$w"
    --test-window "$p"
    --window-mode no_gap
    --tuning-granularity "$tau"
    --train-frame-selection all
    --alpha 1e-6
    --psnr-weight 1
    --ssim-weight 1
    --cr-weight 1
    --quiet-features
    --save-prefix "W${w}_P${p}_tau${tau}"
    --output-dir "$DATASET_ROOT/$dataset/$objective"
  )
  [[ $max_cr == none ]]||command+=(--max-cr "$max_cr")

  echo "============================================================"
  echo "[$order/$TOTAL] Dataset=$dataset Objective=$objective W/P=$w/$p tau=$tau"
  echo "Paper: $paper"
  echo "Output: $result"
  echo "============================================================"
  "${command[@]}"

  [[ -d $result ]]||{ echo "Model did not create $result" >&2;exit 1;}
  for file in compressor_timing.csv joint_prediction_accuracy.csv joint_selection_comparison.csv run_summary.log;do
    [[ -f $result/$file ]]||{ echo "Missing model output: $result/$file" >&2;exit 1;}
  done
  extras=$(find "$result" -maxdepth 1 -type f \
    ! -name compressor_timing.csv \
    ! -name joint_prediction_accuracy.csv \
    ! -name joint_selection_comparison.csv \
    ! -name run_summary.log -print)
  [[ -z $extras ]]||{ echo "Unexpected files in $result:" >&2;echo "$extras" >&2;exit 1;}
  "$PYTHON" "$COMPARE" "$result" "$ROOT/$reference"
done < "$MANIFEST"

if ((RUN_NOTEBOOK));then
  tmp_notebook_dir=$(mktemp -d "${TMPDIR:-/tmp}/pilot-paper-notebook.XXXXXX")
  trap 'rm -rf -- "$tmp_notebook_dir"' EXIT
  PILOT_OUTPUT_ROOT=$OUTPUT_ROOT "$JUPYTER" nbconvert --to notebook --execute "$NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --output reproduce_paper_results_executed.ipynb \
    --output-dir "$tmp_notebook_dir"
  rm -rf -- "$tmp_notebook_dir"
  trap - EXIT
fi

OVERALL_END_NS=$(date +%s%N)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
WALL_TIME_S=$("$PYTHON" -c 'import sys;print(f"{(int(sys.argv[2])-int(sys.argv[1]))/1e9:.6f}")' "$OVERALL_START_NS" "$OVERALL_END_NS")
cat > "$TIMING_FILE" <<EOF
start_utc=$START_UTC
end_utc=$END_UTC
wall_time_s=$WALL_TIME_S
experiments=$TOTAL
execution=sequential
includes=model_runs,reference_validation,notebook_postprocessing
EOF

echo
echo "All $TOTAL paper experiments completed and matched their references."
echo "Overall artifact execution time: $WALL_TIME_S s"
echo "Results: $DATASET_ROOT"
echo "AD timing: $TIMING_FILE"
