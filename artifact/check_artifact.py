#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

INPUTS={
    "CSSI":("lightsource_data_cmp_results/CSSI/cssi_results_600_relEB-2-6-10.csv",36000,"4f942178b2cc8c62a201da81cb6fc64e76d00e2e9c7bd1e055f3a97d79f382f9"),
    "FF-HEDM":("lightsource_data_cmp_results/REI/rei_results_1440_relEB-2-6-10.csv",86400,"701e64fbe228d7cc515a5518f168a820fe83f9ed318006f717d7824c20cbe884"),
    "SFC-GI":("lightsource_data_cmp_results/SFC-GI/sfc-gi_758_relEB-6-2-10.csv",45480,"94d2ac3f2860470e85c6444ff6a7b9d5b5cd219b5e02f46ad7596a4f5c9d5c80"),
    "SFC-L":("lightsource_data_cmp_results/SFC-L/sfc-l_results_10000_relEB-2-6-10.csv",600000,"4ef345bf24523a954a8fd9a6ddb061b25caa03ba84b02df1deda536811f5896a"),
    "XPCS":("lightsource_data_cmp_results/XPCS/xpcs_results_512_relEB-6-2-10.csv",30720,"71e320ec663b8e8d1e7093c05abacd62118b1bd389d78ae5701449b38170bf81"),
}

FIGURE_INPUTS={
    "CSSI frame-32 ROI":("lightsource_data_cmp_results/CSSI/figure_inputs/cssi_frame32_roi_128_rel_50.csv",300,"8181d57f83fc176801e409c44ac25f4dce7399e61c9e49cb5067c3cf9f39df2c"),
}

BASE_COLUMNS={"frame","compressor_id","rel_bound","cr","psnr","ssim_pressio","compression_time_ms","total_feature_time_ms"}
REFERENCE_FILES=("run_summary.log",)
MANIFEST=ROOT/"artifact"/"paper_experiment_manifest.csv"
REQUIRED_FILES=(
    ROOT/"model"/"multi_metric_joint_model.py",
    ROOT/"model"/"reproduce_paper_results.ipynb",
    ROOT/"artifact"/"setup_env.sh",
    ROOT/"artifact"/"run_all_paper_experiments.sh",
    MANIFEST,
)

def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def data_rows(path:Path)->int:
    with path.open("rb") as stream:return max(sum(1 for _ in stream)-1,0)

def main()->None:
    parser=argparse.ArgumentParser(description="Validate PILOT artifact inputs and committed references.")
    parser.add_argument("--skip-checksums",action="store_true",help="Skip SHA-256 verification of the five paper input CSVs.")
    args=parser.parse_args()
    for name,(relative,expected_rows,expected_hash) in INPUTS.items():
        path=ROOT/relative
        if not path.is_file():raise FileNotFoundError(f"{name}: {path}")
        rows=data_rows(path)
        columns={column.strip() for column in pd.read_csv(path,nrows=0,skipinitialspace=True).columns}
        missing=sorted(BASE_COLUMNS-columns)
        actual_hash=None if args.skip_checksums else sha256(path)
        if rows!=expected_rows:raise ValueError(f"{name}: expected {expected_rows} rows, found {rows}")
        if missing:raise ValueError(f"{name}: missing columns {missing}")
        if actual_hash is not None and actual_hash!=expected_hash:
            raise ValueError(f"{name}: SHA-256 mismatch\nexpected {expected_hash}\nfound    {actual_hash}")
        print(f"[OK] {name}: {rows:,} rows"+("" if args.skip_checksums else f", SHA-256 {actual_hash[:12]}..."))

    for name,(relative,expected_rows,expected_hash) in FIGURE_INPUTS.items():
        path=ROOT/relative
        if not path.is_file():raise FileNotFoundError(f"{name}: {path}")
        rows=data_rows(path);actual_hash=None if args.skip_checksums else sha256(path)
        if rows!=expected_rows:raise ValueError(f"{name}: expected {expected_rows} rows, found {rows}")
        if actual_hash is not None and actual_hash!=expected_hash:
            raise ValueError(f"{name}: SHA-256 mismatch\nexpected {expected_hash}\nfound    {actual_hash}")
        print(f"[OK] {name}: {rows:,} rows"+("" if args.skip_checksums else f", SHA-256 {actual_hash[:12]}..."))

    missing_files=[str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing_files:raise FileNotFoundError(f"Missing artifact files: {missing_files}")

    manifest=pd.read_csv(MANIFEST,keep_default_na=False)
    if len(manifest)!=21:raise ValueError(f"Expected 21 paper experiments, found {len(manifest)}")
    for spec in manifest.itertuples(index=False):
        name=f"{spec.order}: {spec.dataset} {spec.objective} {spec.train_window}/{spec.test_window} tau={spec.tau}"
        relative=spec.reference_path
        directory=ROOT/relative
        missing=[filename for filename in REFERENCE_FILES if not (directory/filename).is_file()]
        if missing:raise FileNotFoundError(f"{name}: missing {missing} in {directory}")
        summary=(directory/"run_summary.log").read_text()
        required_labels=("candidate frames:","application frames:","decision frames:")
        if not all(label in summary for label in required_labels):
            raise ValueError(f"{name}: incomplete reference summary")
        print(f"[OK] {name}: committed reference summary present")

    print("Artifact validation passed.")

if __name__=="__main__":main()
