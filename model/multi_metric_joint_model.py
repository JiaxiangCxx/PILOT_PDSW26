#!/usr/bin/env python3
"""Online joint selection using CR with PSNR, SSIM, or both quality metrics.

Adjustable variables:
    --csv-path: offline CSV containing actual PSNR and CR
    --raw-data-path: original uint16 frame file used for paper CR features
    --raw-shape: raw frame shape as frames,height,width
    --train-window: labeled frames used to train each window
    --test-window: following frames predicted before retraining
    --window-mode: disjoint preserves the original schedule; no_gap keeps the old model active while the next model is trained
    --tuning-granularity: number of consecutive application frames sharing one model decision
    --train-frame-selection: all, random, odd, or even training frames
    --selected-train-count: number selected from random/odd/even candidates
    --random-seed: random frame-selection seed
    --candidate-ebs: comma-separated relative EB candidates; default uses all
    --candidate-compressors: comma-separated compressors; default uses all
    --svd-energy: retained SVD energy for the Underwood spatial feature
    --objective: cr, psnr, ssim, psnr_cr, ssim_cr, or psnr_ssim_cr
    --max-cr: keep candidate rows at or below this CR; default uses all
    --psnr-weight: PSNR weight when PSNR is active
    --ssim-weight: SSIM weight when SSIM is active
    --cr-weight: CR weight; active weights are normalized to sum to one
    --alpha: Ridge regularization strength
    --max-frames: optional first N candidate frames for a small experiment
    --save-prefix: experiment result folder name; omit to print without saving
    --output-dir: directory containing experiment folders

Model:
    only the targets required by --objective are trained per compressor
    PSNR inputs: log1p(frame range) + log10(relative EB)
    SSIM inputs: log1p(frame range) + frame mean
                 + log1p(frame variance) + log10(relative EB)
    SSIM target: log(1 - SSIM + epsilon)
    Underwood CR inputs: log(quantized entropy) + log(SVD-trunc/std)
                         + their interaction + log10(relative EB)
    New compression CSVs provide data_std, quantized_entropy,
    svd_trunc_fraction, and total_feature_time_ms directly.
    Raw-data feature calculation remains available only as a fallback for old CSVs.

Example:
    cd /path/to/PDSW_26
    python model/multi_metric_joint_model.py \
      --csv-path lightsource_data_cmp_results/SFC-L/sfc-l_results_10000_relEB-2-6-10.csv \
      --objective psnr_ssim_cr \
      --train-window 16 \
      --test-window 16 \
      --window-mode no_gap \
      --tuning-granularity 1 \
      --train-frame-selection all \
      --max-cr 500 \
      --psnr-weight 1 \
      --ssim-weight 1 \
      --cr-weight 1 \
      --save-prefix psnr_ssim_cr_train16_test16 \
      --output-dir artifact/output

Random/even/odd example:
    python model/multi_metric_joint_model.py \
      --objective ssim_cr \
      --train-window 16 \
      --test-window 16 \
      --train-frame-selection random \
      --selected-train-count 4 \
      --random-seed 0 \
      --save-prefix ssim_cr_random4_train16_test16
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable

# Prevent shared Spack Python packages from overriding this script's conda packages.
sys.path[:] = [entry for entry in sys.path if "spack" not in entry.lower()]
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = str(PROJECT_ROOT / "lightsource_data_cmp_results" / "SFC-L" / "sfc-l_results_10000_relEB-2-6-10.csv")
DEFAULT_RAW_PATH = str(PROJECT_ROOT / "artifact" / "raw_data" / "0-sfc.uint16")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "artifact" / "output" / "model_runs")
EPS = 1e-12
SSIM_EPS = 1e-6
OBJECTIVE_METRICS = {
    "cr": ("cr",),
    "psnr": ("psnr",),
    "ssim": ("ssim",),
    "psnr_ssim": ("psnr", "ssim"),
    "psnr_cr": ("psnr", "cr"),
    "ssim_cr": ("ssim", "cr"),
    "psnr_ssim_cr": ("psnr", "ssim", "cr"),
}
METRIC_COLUMNS = {
    "psnr": ("actual_psnr", "pred_psnr"),
    "ssim": ("actual_ssim", "pred_ssim"),
    "cr": ("actual_cr", "pred_cr"),
}

def parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]

def parse_str_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(item.strip()) for item in value.split(","))
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise argparse.ArgumentTypeError("raw-shape must be frames,height,width with positive integers.")
    return shape

def safe_experiment_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    cleaned = "".join(char if char in allowed else "_" for char in name.strip()).strip("._-")
    if not cleaned:
        raise ValueError("--save-prefix cannot be empty.")
    return cleaned

def load_metric_csv(path: str | Path, replace_nonfinite: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if "compressor_id" not in df.columns:
        raise ValueError("CSV must contain compressor_id.")
    df["compressor_id"] = df["compressor_id"].astype(str).str.strip()
    for column in df.columns:
        if column != "compressor_id":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan) if replace_nonfinite else df

def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.candidate_compressors:
        out = out[out["compressor_id"].isin(args.candidate_compressors)].copy()
    if args.candidate_ebs:
        keep = np.zeros(len(out), dtype=bool)
        for eb in args.candidate_ebs:
            keep |= np.isclose(out["rel_bound"], eb, rtol=0, atol=1e-12)
        out = out[keep].copy()
    frames = np.array(sorted(out["frame"].dropna().astype(int).unique()))
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
        out = out[out["frame"].isin(frames)].copy()
    return out

def filter_max_cr(df: pd.DataFrame, max_cr: float | None) -> tuple[pd.DataFrame, dict[str, object]]:
    before = len(df)
    if max_cr is None:
        return df.copy(), {"max_cr": None, "rows_before": before, "rows_removed": 0, "rows_after": before}
    # CR values above the requested limit are excluded from training, prediction, and oracle selection.
    out = df[df["cr"].isna() | df["cr"].le(max_cr)].copy()
    return out, {"max_cr": max_cr, "rows_before": before, "rows_removed": before - len(out), "rows_after": len(out)}

def prepare_base_features(df: pd.DataFrame, active_metrics: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    out["data_range"] = out["frame_range"]
    required = ["frame", "compressor_id", "rel_bound", "cr", "data_range", "data_variance"]
    if "psnr" in active_metrics:
        required.append("psnr")
    if "ssim" in active_metrics:
        required.extend(["ssim_pressio", "data_mean"])
    out = out.dropna(subset=required).copy()
    out = out[(out["data_range"] > 0) & (out["rel_bound"] > 0) & (out["cr"] > 0)].copy()
    if "ssim" in active_metrics:
        out = out[out["ssim_pressio"].between(0.0, 1.0, inclusive="both")].copy()
    out["frame"] = out["frame"].astype(int)
    # New CSVs store data_std during compression; old CSVs derive it from data_variance.
    if "data_std" not in out.columns:
        out["data_std"] = np.sqrt(out["data_variance"].clip(lower=0))
    out["log_data_range"] = np.log1p(out["data_range"])
    out["log_data_variance"] = np.log1p(out["data_variance"].clip(lower=0))
    out["log_data_std"] = np.log1p(out["data_std"])
    out["log_rel_bound"] = np.log10(out["rel_bound"])
    out["log_abs_bound"] = np.log10((out["data_range"] * out["rel_bound"]).clip(lower=EPS))
    out["log_cr"] = np.log(out["cr"].clip(lower=EPS))
    if "ssim" in active_metrics:
        out["log_ssim_loss"] = np.log((1.0 - out["ssim_pressio"] + SSIM_EPS).clip(lower=SSIM_EPS))
    final_required = ["log_cr", "log_data_range", "log_data_variance", "log_data_std", "log_rel_bound", "log_abs_bound"]
    if "psnr" in active_metrics:
        final_required.append("psnr")
    if "ssim" in active_metrics:
        final_required.extend(["data_mean", "log_ssim_loss"])
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=final_required).copy()

def quantized_entropy(frame: np.ndarray, abs_bound: float) -> float:
    if not np.isfinite(abs_bound) or abs_bound <= 0:
        return np.nan
    codes = np.floor((frame - frame.min()) / abs_bound).astype(np.int64)
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    # Shannon Entropy small means more compressed
    return float(-(probabilities * np.log2(probabilities)).sum())

def svd_trunc_fraction(frame: np.ndarray, energy: float) -> float:
    centered = frame.astype(np.float64) - float(np.mean(frame))
    scale = np.linalg.norm(centered)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    singular_values = np.linalg.svd(centered / scale, compute_uv=False)
    squared = singular_values ** 2
    total = squared.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    retained = int(np.searchsorted(np.cumsum(squared), energy * total) + 1)
    return retained / len(singular_values)

def add_features(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, float, str]:
    # Prefer features recorded by the compression run so model evaluation does not repeat SVD/entropy.
    # These columns are valid model inputs because they are computed from the original frame and candidate EB.
    csv_feature_cols = ["quantized_entropy", "svd_trunc_fraction", "data_std"]
    if all(column in df.columns for column in csv_feature_cols):
        out = df.copy()
        out["raw_data_std"] = out["data_std"]
        if not out["svd_trunc_fraction"].notna().any():
            raise ValueError(
                "CSV contains svd_trunc_fraction but all values are missing. "
                "Recompute this per-frame feature before running the model."
            )
        out["log_quantized_entropy"] = np.log(out["quantized_entropy"].clip(lower=EPS))
        valid_std = out["raw_data_std"].where(out["raw_data_std"] > 0)
        ratio = out["svd_trunc_fraction"] / valid_std
        out["log_svd_std_ratio"] = np.log(ratio.clip(lower=EPS))
        out["entropy_x_svd"] = out["log_quantized_entropy"] * out["log_svd_std_ratio"]
        feature_cols = ["log_quantized_entropy", "log_svd_std_ratio", "entropy_x_svd"]
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy()

        # total_feature_time_ms is repeated for compressor/EB rows, so count it once per frame.
        feature_time_ms = 0.0
        if "total_feature_time_ms" in out.columns:
            feature_time_ms = float(out.groupby("frame")["total_feature_time_ms"].first().sum())
        return out, feature_time_ms, "CSV precomputed columns"

    # Older CSV fallback: read the original frames and calculate the paper features here.
    raw_path = Path(args.raw_data_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    expected_bytes = int(np.prod(args.raw_shape)) * np.dtype(np.uint16).itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(f"Raw file size does not match --raw-shape {args.raw_shape}.")
    raw = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=args.raw_shape)
    frame_ids = np.array(sorted(df["frame"].unique()), dtype=int)
    if frame_ids.max() >= args.raw_shape[0]:
        raise ValueError("CSV frame index exceeds the raw-data frame count.")
    rel_bounds = np.array(sorted(df["rel_bound"].unique()), dtype=float)
    rows = []
    start = perf_counter()
    for position, frame_id in enumerate(frame_ids, start=1):
        frame = np.asarray(raw[frame_id], dtype=np.float64)
        data_range = float(np.ptp(frame))
        data_std = float(np.std(frame))
        svd_fraction = svd_trunc_fraction(frame, args.svd_energy)
        for rel_bound in rel_bounds:
            rows.append({
                "frame": frame_id,
                "rel_bound": rel_bound,
                "quantized_entropy": quantized_entropy(frame, rel_bound * data_range),
                "svd_trunc_fraction": svd_fraction,
                "raw_data_std": data_std,
            })
        if not args.quiet_features and (position % 100 == 0 or position == len(frame_ids)):
            print(f"paper features: {position}/{len(frame_ids)} frames")
    feature_time_ms = (perf_counter() - start) * 1000.0
    features = pd.DataFrame(rows)
    features["log_quantized_entropy"] = np.log(features["quantized_entropy"].clip(lower=EPS))
    ratio = features["svd_trunc_fraction"] / features["raw_data_std"].clip(lower=EPS)
    features["log_svd_std_ratio"] = np.log(ratio.clip(lower=EPS))
    features["entropy_x_svd"] = features["log_quantized_entropy"] * features["log_svd_std_ratio"]
    out = df.merge(features, on=["frame", "rel_bound"], how="left", validate="many_to_one")
    feature_cols = ["log_quantized_entropy", "log_svd_std_ratio", "entropy_x_svd"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy()
    return out, feature_time_ms, "raw-data fallback"

def build_window_specs(frames: np.ndarray, train_window: int, test_window: int, window_mode: str = "disjoint") -> list[dict[str, object]]:
    if train_window < 1 or test_window < 1:
        raise ValueError("train-window and test-window must be positive.")
    if window_mode not in {"disjoint", "no_gap"}:
        raise ValueError("window-mode must be disjoint or no_gap.")
    needed = train_window + test_window
    if len(frames) < needed:
        raise ValueError(f"Need at least {needed} frames, but only {len(frames)} are available.")
    specs = []
    start = 0
    while start + needed <= len(frames):
        specs.append({
            "window_id": len(specs),
            "train_frames": frames[start:start + train_window],
            "test_frames": frames[start + train_window:start + needed],
        })
        start += train_window + test_window
    for index, spec in enumerate(specs):
        carryover = specs[index + 1]["train_frames"] if window_mode == "no_gap" and index + 1 < len(specs) else np.array([], dtype=frames.dtype)
        spec["carryover_frames"] = carryover
        spec["application_frames"] = np.concatenate([spec["test_frames"], carryover])
    return specs

def build_granularity_map(
    spec: dict[str, object],
    tuning_granularity: int,
    next_selected_training_frames: np.ndarray,
) -> pd.DataFrame:
    application_frames = np.asarray(spec["application_frames"], dtype=int)
    primary_frames = set(int(frame) for frame in spec["test_frames"])
    training_window_frames = set(int(frame) for frame in spec["carryover_frames"])
    selected_training_frames = set(int(frame) for frame in next_selected_training_frames)
    rows = []
    for block_index, start in enumerate(range(0, len(application_frames), tuning_granularity)):
        block_frames = application_frames[start:start + tuning_granularity]
        decision_frame = int(block_frames[0])
        for block_offset, application_frame in enumerate(block_frames):
            application_frame = int(application_frame)
            rows.append({
                "window_id": int(spec["window_id"]),
                "model_id": f"M{spec['window_id']}",
                "decision_frame": decision_frame,
                "application_frame": application_frame,
                "tuning_granularity": tuning_granularity,
                "granularity_block_id": f"{spec['window_id']}:{block_index}",
                "block_offset": block_offset,
                "is_decision_frame": block_offset == 0,
                "is_primary_prediction_frame": application_frame in primary_frames,
                "is_training_window_frame": application_frame in training_window_frames,
                "is_selected_training_frame": application_frame in selected_training_frames,
            })
    return pd.DataFrame(rows)

def get_selected_train_count(args: argparse.Namespace, default_count: int) -> int:
    count = args.selected_train_count
    if count is None:
        count = args.random_train_count
    if count is None:
        count = default_count
    if count < 1:
        raise ValueError("selected-train-count must be positive.")
    return count

def evenly_spaced_subset(frames: np.ndarray, count: int) -> np.ndarray:
    if count >= len(frames):
        return frames
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return frames[np.unique(np.maximum.accumulate(indices))]

def select_training_frames(train_frames: np.ndarray, args: argparse.Namespace, window_id: int) -> np.ndarray:
    if args.train_frame_selection == "all":
        return train_frames
    if args.train_frame_selection == "odd":
        candidates = train_frames[(train_frames.astype(int) % 2) == 1]
        selected = evenly_spaced_subset(candidates, get_selected_train_count(args, len(candidates)))
    elif args.train_frame_selection == "even":
        candidates = train_frames[(train_frames.astype(int) % 2) == 0]
        selected = evenly_spaced_subset(candidates, get_selected_train_count(args, len(candidates)))
    elif args.train_frame_selection == "random":
        count = min(get_selected_train_count(args, len(train_frames)), len(train_frames))
        rng = np.random.default_rng(args.random_seed + window_id)
        selected = np.sort(rng.choice(train_frames, size=count, replace=False))
    else:
        raise ValueError("train-frame-selection must be all, random, odd, or even.")
    if len(selected) == 0:
        raise ValueError(f"No training frames selected for window {window_id}.")
    return selected

def fit_predict_target(train: pd.DataFrame, test: pd.DataFrame, compressors: list[str], feature_cols: list[str], target_col: str, target_name: str, alpha: float) -> tuple[pd.Series, float, float, list[dict[str, object]], dict[str, dict[str, object]]]:
    prediction = pd.Series(index=test.index, dtype=float)
    total_train_ms = 0.0
    total_predict_ms = 0.0
    timing_rows = []
    fitted_models = {}
    for compressor in compressors:
        train_comp = train[train["compressor_id"].eq(compressor)].dropna(subset=feature_cols + [target_col])
        test_comp = test[test["compressor_id"].eq(compressor)].dropna(subset=feature_cols)
        row = {"target": target_name, "compressor": compressor, "train_rows": len(train_comp), "test_rows": len(test_comp), "training_time_ms": 0.0, "prediction_time_ms": 0.0, "skipped": False, "skip_reason": ""}
        if test_comp.empty:
            row.update(skipped=True, skip_reason="no_test_rows")
            timing_rows.append(row)
            continue
        if len(train_comp) < 2:
            row.update(skipped=True, skip_reason="insufficient_training_rows")
            timing_rows.append(row)
            continue
        start = perf_counter()
        mean = train_comp[feature_cols].mean()
        std = train_comp[feature_cols].std().mask(lambda x: x.abs() < EPS, 1.0).fillna(1.0)
        x_train = np.c_[np.ones(len(train_comp)), ((train_comp[feature_cols] - mean) / std).to_numpy(float)]
        y_train = train_comp[target_col].to_numpy(float)
        identity = np.eye(x_train.shape[1])
        identity[0, 0] = 0
        matrix = x_train.T @ x_train + alpha * identity
        rhs = x_train.T @ y_train
        try:
            beta = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        train_ms = (perf_counter() - start) * 1000.0
        fitted_models[compressor] = {
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "beta": beta,
        }
        start = perf_counter()
        x_test = np.c_[np.ones(len(test_comp)), ((test_comp[feature_cols] - mean) / std).to_numpy(float)]
        values = x_test @ beta
        predict_ms = (perf_counter() - start) * 1000.0
        row.update(training_time_ms=train_ms, prediction_time_ms=predict_ms)
        total_train_ms += train_ms
        total_predict_ms += predict_ms
        if not np.isfinite(values).all():
            row.update(skipped=True, skip_reason="nonfinite_prediction")
            timing_rows.append(row)
            continue
        prediction.loc[test_comp.index] = values
        timing_rows.append(row)
    return prediction.dropna(), total_train_ms, total_predict_ms, timing_rows, fitted_models

def predict_fitted_target(test: pd.DataFrame, fitted_models: dict[str, dict[str, object]]) -> tuple[pd.Series, float, dict[str, dict[str, float]]]:
    prediction = pd.Series(index=test.index, dtype=float)
    total_predict_ms = 0.0
    timing = {}
    for compressor, model in fitted_models.items():
        feature_cols = model["feature_cols"]
        test_comp = test[test["compressor_id"].eq(compressor)].dropna(subset=feature_cols)
        row = {"test_rows": len(test_comp), "prediction_time_ms": 0.0}
        if test_comp.empty:
            timing[compressor] = row
            continue
        x_test = np.c_[
            np.ones(len(test_comp)),
            ((test_comp[feature_cols] - model["mean"]) / model["std"]).to_numpy(float),
        ]
        start = perf_counter()
        values = x_test @ model["beta"]
        predict_ms = (perf_counter() - start) * 1000.0
        row["prediction_time_ms"] = predict_ms
        total_predict_ms += predict_ms
        if np.isfinite(values).all():
            prediction.loc[test_comp.index] = values
        timing[compressor] = row
    return prediction.dropna(), total_predict_ms, timing

def prepare_inference_features(source: pd.DataFrame, prepared: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = source.copy()
    out["frame"] = out["frame"].astype(int)
    out["data_range"] = out["frame_range"]
    if "data_std" not in out.columns:
        out["data_std"] = np.sqrt(out["data_variance"].clip(lower=0))
    out["raw_data_std"] = out["data_std"]
    out["log_data_range"] = np.log1p(out["data_range"])
    out["log_data_variance"] = np.log1p(out["data_variance"].clip(lower=0))
    out["log_rel_bound"] = np.log10(out["rel_bound"].where(out["rel_bound"] > 0))
    if {"quantized_entropy", "svd_trunc_fraction"}.issubset(out.columns):
        out["log_quantized_entropy"] = np.log(out["quantized_entropy"].clip(lower=EPS))
        ratio = out["svd_trunc_fraction"] / out["raw_data_std"].where(out["raw_data_std"] > 0)
        out["log_svd_std_ratio"] = np.log(ratio.clip(lower=EPS))
        out["entropy_x_svd"] = out["log_quantized_entropy"] * out["log_svd_std_ratio"]
    missing = [column for column in feature_cols if column not in out.columns or not out[column].notna().any()]
    if missing:
        lookup = prepared[["frame", "rel_bound"] + missing].dropna(subset=missing).drop_duplicates(
            ["frame", "rel_bound"], keep="last"
        )
        out = out.drop(columns=[column for column in missing if column in out.columns]).merge(
            lookup, on=["frame", "rel_bound"], how="left", validate="many_to_one"
        )
    require_columns(out, feature_cols)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.drop_duplicates(["frame", "compressor_id", "rel_bound"], keep="last")

def group_minmax(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= EPS:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)

def finite_sum(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.sum()) if len(values) else np.nan

def normalized_active_weights(args: argparse.Namespace, active_metrics: tuple[str, ...]) -> dict[str, float]:
    raw = {"psnr": args.psnr_weight, "ssim": args.ssim_weight, "cr": args.cr_weight}
    selected = {metric: raw[metric] for metric in active_metrics}
    if any(weight < 0 for weight in selected.values()) or sum(selected.values()) <= 0:
        raise ValueError("Weights for active metrics must be nonnegative and cannot all be zero.")
    total = sum(selected.values())
    return {metric: weight / total for metric, weight in selected.items()}

def add_prediction_accuracy(df: pd.DataFrame, active_metrics: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for metric in active_metrics:
        actual_col, pred_col = METRIC_COLUMNS[metric]
        out[f"{metric}_err"] = (out[pred_col] - out[actual_col]).abs()
        denominator = out[actual_col].abs()
        out[f"{metric}_rel_err"] = np.where(denominator > EPS, out[f"{metric}_err"] / denominator, np.nan)
        out[f"{metric}_accuracy"] = np.clip(100.0 * (1.0 - out[f"{metric}_rel_err"]), 0.0, 100.0)
    return out

def add_joint_score(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
    active_metrics: tuple[str, ...],
    score_kind: str,
) -> pd.DataFrame:
    if score_kind not in {"pred", "actual"}:
        raise ValueError("score-kind must be pred or actual.")
    out = predictions.copy()
    weights = normalized_active_weights(args, active_metrics)
    score_col = f"{score_kind}_joint_score"
    out[score_col] = 1.0
    for metric in active_metrics:
        actual_col, pred_col = METRIC_COLUMNS[metric]
        value_col = pred_col if score_kind == "pred" else actual_col
        norm_col = f"{score_kind}_{metric}_norm"
        if metric == "cr":
            out[norm_col] = out.groupby("frame")[value_col].transform(lambda x: group_minmax(np.log1p(x)))
        elif metric == "ssim":
            # SSIM already has a meaningful [0, 1] scale; per-frame min-max would amplify tiny ties near 1.
            out[norm_col] = out[value_col].clip(0.0, 1.0)
        else:
            out[norm_col] = out.groupby("frame")[value_col].transform(group_minmax)
        out[score_col] *= (EPS + out[norm_col]) ** weights[metric]
    return out

def add_joint_scores(predictions: pd.DataFrame, args: argparse.Namespace, active_metrics: tuple[str, ...]) -> pd.DataFrame:
    out = add_joint_score(predictions, args, active_metrics, "pred")
    return add_joint_score(out, args, active_metrics, "actual")

def evaluate_selection(
    predictions: pd.DataFrame,
    active_metrics: tuple[str, ...],
    predicted_best_index: pd.Index | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    predicted_best = (
        predictions.loc[predicted_best_index]
        if predicted_best_index is not None
        else predictions.sort_values(["frame", "pred_joint_score"], ascending=[True, False]).groupby("frame").head(1)
    )
    oracle_best = predictions.sort_values(["frame", "actual_joint_score"], ascending=[True, False]).groupby("frame").head(1)
    metadata = ["window_id", "train_frame_start", "train_frame_end", "test_frame_start", "test_frame_end", "training_frame_count", "frame"]
    metric_cols = [column for metric in active_metrics for column in reversed(METRIC_COLUMNS[metric])]
    runtime_cols = ["compression_time_ms"] if "compression_time_ms" in predicted_best.columns else []
    selected = predicted_best[metadata + ["compressor", "eb"] + metric_cols + runtime_cols + ["pred_joint_score", "actual_joint_score"]].rename(columns={
        "compressor": "pred_compressor", "eb": "pred_eb", "actual_joint_score": "actual_joint_score_of_pred_choice"
    })
    oracle_actual = [METRIC_COLUMNS[metric][0] for metric in active_metrics]
    rename_oracle = {"compressor": "oracle_compressor", "eb": "oracle_eb", "actual_joint_score": "oracle_joint_score"}
    rename_oracle.update({METRIC_COLUMNS[metric][0]: f"oracle_{metric}" for metric in active_metrics})
    oracle = oracle_best[["frame", "compressor", "eb"] + oracle_actual + ["actual_joint_score"]].rename(columns=rename_oracle)
    selected = selected.merge(oracle, on="frame", validate="one_to_one")
    selected["compressor_correct"] = selected["pred_compressor"].eq(selected["oracle_compressor"])
    selected["eb_correct"] = np.isclose(selected["pred_eb"], selected["oracle_eb"], rtol=0, atol=1e-12)
    selected["pair_correct"] = selected["compressor_correct"] & selected["eb_correct"]
    selected = add_prediction_accuracy(selected, active_metrics)
    selected["relative_score_gap_pct"] = (selected["oracle_joint_score"] - selected["actual_joint_score_of_pred_choice"]) / selected["oracle_joint_score"].abs().clip(lower=EPS) * 100.0
    summary = {
        "pair_correct_pct": 100.0 * selected["pair_correct"].mean(),
        "compressor_correct_pct": 100.0 * selected["compressor_correct"].mean(),
        "eb_correct_pct": 100.0 * selected["eb_correct"].mean(),
        "mean_relative_score_gap_pct": selected["relative_score_gap_pct"].mean(),
    }
    for metric in active_metrics:
        actual_col = METRIC_COLUMNS[metric][0]
        gap_col = "psnr_gap_dB" if metric == "psnr" else f"{metric}_gap"
        gap = selected[f"oracle_{metric}"] - selected[actual_col]
        selected[gap_col] = gap.where(
            np.isfinite(selected[f"oracle_{metric}"]) & np.isfinite(selected[actual_col])
        )
        summary[f"selected_mean_{metric}_accuracy_pct"] = selected[f"{metric}_accuracy"].mean()
        summary[f"mean_{gap_col}"] = selected[gap_col].mean()
    return selected, summary

def build_application_selection(
    decision_selection: pd.DataFrame,
    application_map: pd.DataFrame,
    candidate_data: pd.DataFrame,
    runtime_data: pd.DataFrame,
    args: argparse.Namespace,
    active_metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    decision_columns = ["window_id", "frame", "pred_compressor", "pred_eb", "pred_joint_score"]
    decision_columns.extend(METRIC_COLUMNS[metric][1] for metric in active_metrics)
    decision = decision_selection[decision_columns].rename(columns={
        "frame": "decision_frame",
        "pred_joint_score": "decision_pred_joint_score",
        **{METRIC_COLUMNS[metric][1]: f"decision_{METRIC_COLUMNS[metric][1]}" for metric in active_metrics},
    })
    selected = application_map.merge(decision, on=["window_id", "decision_frame"], how="left", validate="many_to_one")
    if selected["pred_compressor"].isna().any() or selected["pred_eb"].isna().any():
        raise ValueError("One or more granularity blocks are missing a model decision.")
    selected["frame"] = selected["application_frame"]
    # A granularity block reuses both the selected pair and its decision-frame
    # predictions. Non-decision frames therefore require no online features or
    # additional model inference.
    for metric in active_metrics:
        pred_col = METRIC_COLUMNS[metric][1]
        selected[pred_col] = selected[f"decision_{pred_col}"]

    source_columns = ["frame", "compressor_id", "rel_bound"]
    source_columns.extend({"psnr": "psnr", "ssim": "ssim_pressio", "cr": "cr"}[metric] for metric in active_metrics)
    candidates = candidate_data[source_columns].drop_duplicates(["frame", "compressor_id", "rel_bound"], keep="last").rename(columns={
        "frame": "application_frame",
        "compressor_id": "candidate_compressor",
        "rel_bound": "candidate_eb",
        "psnr": "actual_psnr",
        "ssim_pressio": "actual_ssim",
        "cr": "actual_cr",
    })
    score_candidates = candidates.rename(columns={"application_frame": "frame"})
    score_candidates = add_joint_score(score_candidates, args, active_metrics, "actual").rename(columns={"frame": "application_frame"})
    chosen_columns = ["application_frame", "candidate_compressor", "candidate_eb", "actual_joint_score"]
    chosen_columns.extend(METRIC_COLUMNS[metric][0] for metric in active_metrics)
    selected = selected.merge(
        score_candidates[chosen_columns],
        left_on=["application_frame", "pred_compressor", "pred_eb"],
        right_on=["application_frame", "candidate_compressor", "candidate_eb"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["candidate_compressor", "candidate_eb"])
    selected = selected.rename(columns={"actual_joint_score": "actual_joint_score_of_pred_choice"})

    oracle = score_candidates.sort_values(["application_frame", "actual_joint_score"], ascending=[True, False]).groupby("application_frame").head(1)
    oracle_columns = ["application_frame", "candidate_compressor", "candidate_eb", "actual_joint_score"]
    oracle_columns.extend(METRIC_COLUMNS[metric][0] for metric in active_metrics)
    oracle = oracle[oracle_columns].rename(columns={
        "candidate_compressor": "oracle_compressor",
        "candidate_eb": "oracle_eb",
        "actual_joint_score": "oracle_joint_score",
        **{METRIC_COLUMNS[metric][0]: f"oracle_{metric}" for metric in active_metrics},
    })
    selected = selected.merge(oracle, on="application_frame", how="left", validate="many_to_one")

    if "compression_time_ms" in runtime_data.columns:
        runtime_columns = ["frame", "compressor_id", "rel_bound", "compression_time_ms"]
        runtime_columns.extend({"psnr": "psnr", "ssim": "ssim_pressio", "cr": "cr"}[metric] for metric in active_metrics)
        runtime_lookup = runtime_data[runtime_columns].drop_duplicates(
            ["frame", "compressor_id", "rel_bound"], keep="last"
        ).rename(columns={
            "frame": "application_frame",
            "compressor_id": "pred_compressor",
            "rel_bound": "pred_eb",
            **{
                {"psnr": "psnr", "ssim": "ssim_pressio", "cr": "cr"}[metric]: f"source_actual_{metric}"
                for metric in active_metrics
            },
        })
        selected = selected.merge(
            runtime_lookup,
            on=["application_frame", "pred_compressor", "pred_eb"],
            how="left",
            validate="many_to_one",
        )
        if selected["compression_time_ms"].isna().any():
            raise ValueError("Missing compression_time_ms for one or more applied compressor/EB choices.")
        for metric in active_metrics:
            actual_col = METRIC_COLUMNS[metric][0]
            selected[actual_col] = selected[f"source_actual_{metric}"]
            selected = selected.drop(columns=f"source_actual_{metric}")

    selected["pred_joint_score"] = selected["decision_pred_joint_score"]
    selected["compressor_correct"] = selected["pred_compressor"].eq(selected["oracle_compressor"])
    selected["eb_correct"] = np.isclose(selected["pred_eb"], selected["oracle_eb"], rtol=0, atol=1e-12)
    selected["pair_correct"] = selected["compressor_correct"] & selected["eb_correct"]
    selected = add_prediction_accuracy(selected, active_metrics)
    selected["relative_score_gap_pct"] = (
        (selected["oracle_joint_score"] - selected["actual_joint_score_of_pred_choice"])
        / selected["oracle_joint_score"].abs().clip(lower=EPS)
        * 100.0
    )

    summary = {
        "pair_correct_pct": 100.0 * selected["pair_correct"].mean(),
        "compressor_correct_pct": 100.0 * selected["compressor_correct"].mean(),
        "eb_correct_pct": 100.0 * selected["eb_correct"].mean(),
        "mean_relative_score_gap_pct": selected["relative_score_gap_pct"].mean(),
        "decision_pair_correct_pct": 100.0 * selected.loc[selected["is_decision_frame"], "pair_correct"].mean(),
        "primary_pair_correct_pct": 100.0 * selected.loc[selected["is_primary_prediction_frame"], "pair_correct"].mean(),
        "carryover_pair_correct_pct": 100.0 * selected.loc[selected["is_training_window_frame"], "pair_correct"].mean(),
        "selection_coverage_pct": 100.0 * len(selected) / len(application_map) if len(application_map) else np.nan,
        "evaluation_coverage_pct": 100.0 * selected["oracle_joint_score"].notna().mean(),
    }
    for metric in active_metrics:
        actual_col = METRIC_COLUMNS[metric][0]
        gap_col = "psnr_gap_dB" if metric == "psnr" else f"{metric}_gap"
        gap = selected[f"oracle_{metric}"] - selected[actual_col]
        selected[gap_col] = gap.where(
            np.isfinite(selected[f"oracle_{metric}"]) & np.isfinite(selected[actual_col])
        )
        summary[f"selected_mean_{metric}_accuracy_pct"] = selected[f"{metric}_accuracy"].mean()
        summary[f"mean_{gap_col}"] = selected[gap_col].mean()
    return selected, summary

def run_window_evaluation(
    data: pd.DataFrame,
    args: argparse.Namespace,
    feature_time_ms: float,
    feature_source: str,
    runtime_data: pd.DataFrame | None = None,
    application_data: pd.DataFrame | None = None,
) -> dict[str, object]:
    active_metrics = OBJECTIVE_METRICS[args.objective]
    runtime_data = data if runtime_data is None else runtime_data
    application_data = runtime_data if application_data is None else application_data
    frames = np.array(sorted(data["frame"].unique()), dtype=int)
    specs = build_window_specs(frames, args.train_window, args.test_window, args.window_mode)
    compressors = sorted(data["compressor_id"].unique())
    model_specs = {
        "psnr": (["log_data_range", "log_rel_bound"], "psnr", "psnr"),
        "ssim": (["log_data_range", "data_mean", "log_data_variance", "log_rel_bound"], "log_ssim_loss", "log_ssim_loss"),
        "cr": (["log_quantized_entropy", "log_svd_std_ratio", "entropy_x_svd", "log_rel_bound"], "log_cr", "log_cr"),
    }
    selected_training_by_window = {
        int(spec["window_id"]): select_training_frames(spec["train_frames"], args, int(spec["window_id"]))
        for spec in specs
    }
    prediction_frames = []
    decision_selection_frames = []
    application_maps = []
    timing_rows = []
    compressor_timing_rows = []
    training_feature_frame_ids = set()
    evaluated_window_ids = set()
    decision_frame_ids = set()
    application_frame_ids = set()
    for spec in specs:
        window_id = int(spec["window_id"])
        train_frames = spec["train_frames"]
        selected_train_frames = selected_training_by_window[window_id]
        next_selected_training_frames = selected_training_by_window.get(window_id + 1, np.array([], dtype=int))
        application_map = build_granularity_map(spec, args.tuning_granularity, next_selected_training_frames)
        application_map["train_frame_start"] = train_frames[0]
        application_map["train_frame_end"] = train_frames[-1]
        application_map["test_frame_start"] = spec["test_frames"][0]
        application_map["test_frame_end"] = spec["test_frames"][-1]
        application_map["training_frame_count"] = len(selected_train_frames)
        decision_frames = application_map.loc[application_map["is_decision_frame"], "application_frame"].to_numpy(dtype=int)
        train = data[data["frame"].isin(selected_train_frames)].copy()
        test = data[data["frame"].isin(decision_frames)].copy()
        predictions_by_metric = {}
        train_ms = 0.0
        predict_ms = 0.0
        model_timing = []
        for metric in active_metrics:
            features, target_col, target_name = model_specs[metric]
            prediction, metric_train_ms, metric_predict_ms, metric_timing, _ = fit_predict_target(
                train, test, compressors, features, target_col, target_name, args.alpha
            )
            predictions_by_metric[metric] = prediction
            train_ms += metric_train_ms
            predict_ms += metric_predict_ms
            model_timing.extend(metric_timing)
        usable_index = test.index
        for prediction in predictions_by_metric.values():
            usable_index = usable_index.intersection(prediction.index)
        if usable_index.empty:
            continue
        source_columns = ["frame", "compressor_id", "rel_bound"]
        source_columns.extend({"psnr": "psnr", "ssim": "ssim_pressio", "cr": "cr"}[metric] for metric in active_metrics)
        if "compression_time_ms" in test.columns:
            source_columns.append("compression_time_ms")
        output = test.loc[usable_index, source_columns].copy().rename(columns={
            "compressor_id": "compressor",
            "rel_bound": "eb",
            "psnr": "actual_psnr",
            "ssim_pressio": "actual_ssim",
            "cr": "actual_cr",
        })
        transform_start = perf_counter()
        if "psnr" in active_metrics:
            output["pred_psnr"] = predictions_by_metric["psnr"].loc[usable_index]
        if "ssim" in active_metrics:
            pred_log_loss = predictions_by_metric["ssim"].loc[usable_index].clip(lower=-30, upper=10)
            output["pred_ssim"] = np.clip(1.0 + SSIM_EPS - np.exp(pred_log_loss), 0.0, 1.0)
        if "cr" in active_metrics:
            output["pred_cr"] = np.exp(predictions_by_metric["cr"].loc[usable_index].clip(lower=-30, upper=30))
        predict_ms += (perf_counter() - transform_start) * 1000.0
        output["window_id"] = window_id
        output["model_id"] = f"M{window_id}"
        output["decision_frame"] = output["frame"]
        output["application_frame"] = output["frame"]
        output["tuning_granularity"] = args.tuning_granularity
        decision_metadata = application_map[application_map["is_decision_frame"]][[
            "decision_frame", "granularity_block_id", "block_offset", "is_decision_frame",
            "is_primary_prediction_frame", "is_training_window_frame", "is_selected_training_frame",
        ]]
        output = output.merge(decision_metadata, on="decision_frame", how="left", validate="many_to_one")
        output["train_frame_start"] = train_frames[0]
        output["train_frame_end"] = train_frames[-1]
        output["test_frame_start"] = spec["test_frames"][0]
        output["test_frame_end"] = spec["test_frames"][-1]
        output["training_frame_count"] = len(selected_train_frames)
        output["training_time_ms"] = train_ms
        selection_start = perf_counter()
        output = add_joint_score(output, args, active_metrics, "pred")
        predicted_best_index = output.sort_values(
            ["frame", "pred_joint_score"], ascending=[True, False]
        ).groupby("frame").head(1).index
        selection_ms = (perf_counter() - selection_start) * 1000.0
        output = add_joint_score(output, args, active_metrics, "actual")
        output = add_prediction_accuracy(output, active_metrics)
        window_decision_selection, _ = evaluate_selection(
            output, active_metrics, predicted_best_index
        )
        output["prediction_time_ms"] = predict_ms
        output["selection_time_ms"] = selection_ms
        prediction_frames.append(output)
        decision_selection_frames.append(window_decision_selection)
        application_maps.append(application_map)
        training_feature_frame_ids.update(int(frame) for frame in selected_train_frames)
        decision_frame_ids.update(int(frame) for frame in decision_frames)
        application_frame_ids.update(int(frame) for frame in spec["application_frames"])
        evaluated_window_ids.add(window_id)
        for row in model_timing:
            row.update({
                "window_id": window_id,
                "model_id": f"M{window_id}",
                "train_frame_start": train_frames[0],
                "train_frame_end": train_frames[-1],
                "test_frame_start": spec["test_frames"][0],
                "test_frame_end": spec["test_frames"][-1],
                "training_frame_count": len(selected_train_frames),
                "decision_frames": len(decision_frames),
                "application_frames": len(spec["application_frames"]),
                "tuning_granularity": args.tuning_granularity,
            })
            compressor_timing_rows.append(row)
        timing_rows.append({
            "window_id": window_id,
            "model_id": f"M{window_id}",
            "training_time_ms": train_ms,
            "prediction_time_ms": predict_ms,
            "selection_time_ms": selection_ms,
            "decision_frames": len(decision_frames),
            "application_frames": len(spec["application_frames"]),
            "predicted_rows": len(output),
            "training_frames": len(selected_train_frames),
            "tuning_granularity": args.tuning_granularity,
        })
    if not prediction_frames:
        raise ValueError("No usable prediction windows were produced.")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    application_map = pd.concat(application_maps, ignore_index=True)
    timing = pd.DataFrame(timing_rows)
    decision_selection = pd.concat(decision_selection_frames, ignore_index=True)
    selection, selection_summary = build_application_selection(
        decision_selection, application_map, data, application_data, args, active_metrics
    )
    expected_application_frames = len(application_frame_ids)
    actual_application_frames = int(selection["application_frame"].nunique())
    if selection["application_frame"].duplicated().any():
        duplicates = selection.loc[
            selection["application_frame"].duplicated(False), "application_frame"
        ].drop_duplicates().tolist()[:10]
        raise ValueError(f"Duplicate application-frame selections detected: {duplicates}")
    if actual_application_frames != expected_application_frames:
        raise ValueError(
            "Application-frame coverage mismatch: "
            f"expected {expected_application_frames}, produced {actual_application_frames}."
        )
    if "cr" in active_metrics:
        valid_selected_cr = np.isfinite(selection["actual_cr"]) & selection["actual_cr"].gt(0)
        if not valid_selected_cr.all():
            bad_frames = selection.loc[
                ~valid_selected_cr, "application_frame"
            ].drop_duplicates().tolist()[:10]
            raise ValueError(f"Invalid selected actual CR for application frames: {bad_frames}")
    total_train_ms = timing["training_time_ms"].sum()
    total_predict_ms = timing["prediction_time_ms"].sum()
    total_selection_ms = timing["selection_time_ms"].sum()
    predicted_frames = int(predictions["frame"].nunique())
    application_frames = int(selection["application_frame"].nunique())
    training_feature_frames = set(training_feature_frame_ids)
    decision_feature_frames = set(decision_frame_ids)
    overlap_feature_frames = training_feature_frames & decision_feature_frames
    end_to_end_feature_frames = training_feature_frames | decision_feature_frames
    training_feature_time_ms = np.nan
    decision_feature_time_ms = np.nan
    overlap_feature_time_ms = np.nan
    end_to_end_unique_feature_time_ms = np.nan
    if "total_feature_time_ms" in runtime_data.columns:
        feature_time_by_frame = runtime_data.groupby("frame")["total_feature_time_ms"].first()

        def feature_time_for(frame_ids: set[int], label: str) -> float:
            if not frame_ids:
                return 0.0
            values = feature_time_by_frame.reindex(sorted(frame_ids))
            if values.isna().any():
                missing = values[values.isna()].index.tolist()[:10]
                raise ValueError(f"Missing total_feature_time_ms for {label} frames: {missing}")
            return finite_sum(values)

        training_feature_time_ms = feature_time_for(training_feature_frames, "training")
        decision_feature_time_ms = feature_time_for(decision_feature_frames, "decision")
        overlap_feature_time_ms = feature_time_for(overlap_feature_frames, "training/decision overlap")
        end_to_end_unique_feature_time_ms = feature_time_for(end_to_end_feature_frames, "end-to-end")
        expected_unique_feature_time_ms = (
            training_feature_time_ms + decision_feature_time_ms - overlap_feature_time_ms
        )
        if not np.isclose(end_to_end_unique_feature_time_ms, expected_unique_feature_time_ms):
            raise ValueError("Training/decision feature-time union accounting is inconsistent.")
    if "compression_time_ms" in predictions.columns and predictions["compression_time_ms"].isna().any():
        raise ValueError("Missing compression_time_ms in one or more predicted candidate rows.")
    baseline_rows = runtime_data[runtime_data["frame"].isin(application_frame_ids)].copy()
    if "compression_time_ms" in baseline_rows.columns and baseline_rows["compression_time_ms"].isna().any():
        raise ValueError("Missing compression_time_ms in one or more baseline trial rows.")
    baseline_all_trials_time_ms = finite_sum(baseline_rows["compression_time_ms"]) if "compression_time_ms" in baseline_rows.columns else np.nan
    selected_compression_time_ms = finite_sum(selection["compression_time_ms"]) if "compression_time_ms" in selection.columns else np.nan
    online_parts = [decision_feature_time_ms, total_predict_ms, total_selection_ms, selected_compression_time_ms]
    online_total_time_ms = float(sum(online_parts)) if all(np.isfinite(value) for value in online_parts) else np.nan
    saved_time_ms = baseline_all_trials_time_ms - online_total_time_ms if np.isfinite(baseline_all_trials_time_ms) and np.isfinite(online_total_time_ms) else np.nan
    time_reduction_pct = 100.0 * saved_time_ms / baseline_all_trials_time_ms if np.isfinite(saved_time_ms) and baseline_all_trials_time_ms > 0 else np.nan
    speedup = baseline_all_trials_time_ms / online_total_time_ms if np.isfinite(baseline_all_trials_time_ms) and np.isfinite(online_total_time_ms) and online_total_time_ms > 0 else np.nan

    training_trial_frames = set()
    for window_id in evaluated_window_ids:
        training_trial_frames.update(int(frame) for frame in selected_training_by_window[window_id])
    training_trial_rows = runtime_data[runtime_data["frame"].isin(training_trial_frames)].copy()
    if "compression_time_ms" in training_trial_rows.columns and training_trial_rows["compression_time_ms"].isna().any():
        raise ValueError("Missing compression_time_ms in one or more training trial rows.")
    training_trial_time_ms = finite_sum(training_trial_rows["compression_time_ms"]) if "compression_time_ms" in training_trial_rows.columns else np.nan
    nontrial_selection = selection[~selection["application_frame"].isin(training_trial_frames)]
    nontrial_selected_compression_time_ms = finite_sum(nontrial_selection["compression_time_ms"]) if "compression_time_ms" in nontrial_selection.columns else np.nan
    end_to_end_frames = application_frame_ids | training_trial_frames
    end_to_end_baseline_rows = runtime_data[runtime_data["frame"].isin(end_to_end_frames)].copy()
    end_to_end_baseline_all_trials_time_ms = finite_sum(end_to_end_baseline_rows["compression_time_ms"]) if "compression_time_ms" in end_to_end_baseline_rows.columns else np.nan
    end_to_end_parts = [
        training_trial_time_ms,
        end_to_end_unique_feature_time_ms,
        total_train_ms,
        total_predict_ms,
        total_selection_ms,
        nontrial_selected_compression_time_ms,
    ]
    end_to_end_online_time_ms = float(sum(end_to_end_parts)) if all(np.isfinite(value) for value in end_to_end_parts) else np.nan
    end_to_end_saved_time_ms = (
        end_to_end_baseline_all_trials_time_ms - end_to_end_online_time_ms
        if np.isfinite(end_to_end_baseline_all_trials_time_ms) and np.isfinite(end_to_end_online_time_ms)
        else np.nan
    )
    end_to_end_time_reduction_pct = (
        100.0 * end_to_end_saved_time_ms / end_to_end_baseline_all_trials_time_ms
        if np.isfinite(end_to_end_saved_time_ms) and end_to_end_baseline_all_trials_time_ms > 0
        else np.nan
    )
    end_to_end_speedup = (
        end_to_end_baseline_all_trials_time_ms / end_to_end_online_time_ms
        if np.isfinite(end_to_end_baseline_all_trials_time_ms)
        and np.isfinite(end_to_end_online_time_ms)
        and end_to_end_online_time_ms > 0
        else np.nan
    )
    overall = {
        "feature_source": feature_source,
        "feature_extraction_time_ms": feature_time_ms,
        "training_feature_extraction_time_ms": training_feature_time_ms,
        "decision_feature_extraction_time_ms": decision_feature_time_ms,
        "overlap_feature_extraction_time_ms": overlap_feature_time_ms,
        "end_to_end_unique_feature_extraction_time_ms": end_to_end_unique_feature_time_ms,
        "total_training_time_ms": total_train_ms,
        "mean_training_time_per_window_ms": timing["training_time_ms"].mean(),
        "total_prediction_time_ms": total_predict_ms,
        "mean_prediction_time_per_window_ms": timing["prediction_time_ms"].mean(),
        "prediction_time_per_decision_ms": total_predict_ms / predicted_frames,
        "throughput_decisions_per_second": predicted_frames / (total_predict_ms / 1000.0) if total_predict_ms > 0 else np.inf,
        "effective_application_throughput_frames_per_second": application_frames / (total_predict_ms / 1000.0) if total_predict_ms > 0 else np.inf,
        "total_selection_time_ms": total_selection_ms,
        "selection_time_per_decision_ms": total_selection_ms / predicted_frames,
    }
    for metric in active_metrics:
        overall[f"mean_{metric}_accuracy_pct"] = predictions[f"{metric}_accuracy"].mean()
    runtime_comparison = {
        "training_feature_frames": len(training_feature_frames),
        "decision_feature_frames": len(decision_feature_frames),
        "overlap_feature_frames": len(overlap_feature_frames),
        "end_to_end_unique_feature_frames": len(end_to_end_feature_frames),
        "training_feature_time_ms": training_feature_time_ms,
        "decision_feature_time_ms": decision_feature_time_ms,
        "overlap_feature_time_ms": overlap_feature_time_ms,
        "end_to_end_unique_feature_time_ms": end_to_end_unique_feature_time_ms,
        "baseline_trial_rows": len(baseline_rows),
        "baseline_all_trials_time_ms": baseline_all_trials_time_ms,
        "model_training_time_ms": float(total_train_ms),
        "model_prediction_time_ms": float(total_predict_ms),
        "model_selection_time_ms": float(total_selection_ms),
        "selected_compression_time_ms": selected_compression_time_ms,
        "online_total_time_ms": online_total_time_ms,
        "saved_time_ms": saved_time_ms,
        "time_reduction_pct": time_reduction_pct,
        "speedup": speedup,
        "training_trial_frames": len(training_trial_frames),
        "training_trial_rows": len(training_trial_rows),
        "training_trial_time_ms": training_trial_time_ms,
        "nontrial_selected_compression_time_ms": nontrial_selected_compression_time_ms,
        "end_to_end_baseline_trial_rows": len(end_to_end_baseline_rows),
        "end_to_end_baseline_all_trials_time_ms": end_to_end_baseline_all_trials_time_ms,
        "end_to_end_online_time_ms": end_to_end_online_time_ms,
        "end_to_end_saved_time_ms": end_to_end_saved_time_ms,
        "end_to_end_time_reduction_pct": end_to_end_time_reduction_pct,
        "end_to_end_speedup": end_to_end_speedup,
    }
    return {
        "predictions": predictions,
        "selection": selection,
        "selection_summary": selection_summary,
        "overall": overall,
        "runtime_comparison": runtime_comparison,
        "timing": timing,
        "compressor_timing": pd.DataFrame(compressor_timing_rows),
        "active_metrics": active_metrics,
        "normalized_weights": normalized_active_weights(args, active_metrics),
        "window_count": len(timing),
        "candidate_frames": int(data["frame"].nunique()),
        "predicted_frames": predicted_frames,
        "application_frames": application_frames,
        "compressors": compressors,
        "error_bounds": sorted(data["rel_bound"].unique()),
    }

def format_summary(args: argparse.Namespace, result: dict[str, object], saved: tuple[Path, Path, Path, Path] | None = None) -> str:
    overall = result["overall"]
    selection = result["selection_summary"]
    runtime = result["runtime_comparison"]
    active_metrics = result["active_metrics"]
    weights = result["normalized_weights"]
    cr_filter = result["cr_filter_summary"]
    lines = [
        "=== Experiment Setup ===", f"dataset: {Path(args.csv_path).resolve()}", f"raw data fallback: {Path(args.raw_data_path).resolve()}",
        f"objective: {args.objective}", f"active metrics: {', '.join(active_metrics)}",
        "model: one Ridge per active target and compressor", f"feature source: {overall['feature_source']}", f"alpha: {args.alpha:g}",
        f"train window: {args.train_window} frames", f"test window: {args.test_window} frames", f"train frame selection: {args.train_frame_selection}",
        f"window mode: {args.window_mode}", f"tuning granularity: {args.tuning_granularity} frames/decision",
        f"selected train count: {args.selected_train_count if args.selected_train_count is not None else 'all available'}",
        f"maximum CR: {cr_filter['max_cr'] if cr_filter['max_cr'] is not None else 'no limit'}",
        f"candidate rows before max-CR filter: {cr_filter['rows_before']}", f"rows removed by max-CR filter: {cr_filter['rows_removed']}", f"candidate rows after max-CR filter: {cr_filter['rows_after']}",
        "normalized weights: " + ", ".join(f"{metric}={weights[metric]:.6f}" for metric in active_metrics),
        f"evaluated windows: {result['window_count']}", f"candidate frames: {result['candidate_frames']}", f"predicted frames: {result['predicted_frames']}", f"decision frames: {result['predicted_frames']}", f"application frames: {result['application_frames']}",
        f"compressors: {len(result['compressors'])}", f"error bounds: {len(result['error_bounds'])}", "", "=== All Candidate Prediction Accuracy ===",
    ]
    for metric in active_metrics:
        lines.append(f"mean {metric.upper()} accuracy: {overall[f'mean_{metric}_accuracy_pct']:.6f}%")
    lines.extend(["", "=== Selected Pair Prediction Accuracy ==="])
    for metric in active_metrics:
        lines.append(f"selected mean {metric.upper()} accuracy: {selection[f'selected_mean_{metric}_accuracy_pct']:.6f}%")
    lines.extend([
        "", "=== Joint Selection Accuracy ===",
        f"compressor + EB correct: {selection['pair_correct_pct']:.2f}%",
        f"decision-frame compressor + EB correct: {selection['decision_pair_correct_pct']:.2f}%",
        f"primary-window compressor + EB correct: {selection['primary_pair_correct_pct']:.2f}%",
        f"carryover-window compressor + EB correct: {selection['carryover_pair_correct_pct']:.2f}%",
        f"compressor correct: {selection['compressor_correct_pct']:.2f}%",
        f"EB correct: {selection['eb_correct_pct']:.2f}%",
        f"selection coverage: {selection['selection_coverage_pct']:.2f}%",
        f"evaluation coverage: {selection['evaluation_coverage_pct']:.2f}%",
        f"mean relative joint-score gap: {selection['mean_relative_score_gap_pct']:.6f}%",
    ])
    for metric in active_metrics:
        gap_key = "mean_psnr_gap_dB" if metric == "psnr" else f"mean_{metric}_gap"
        suffix = " dB" if metric == "psnr" else ""
        lines.append(f"mean {metric.upper()} gap: {selection[gap_key]:.6f}{suffix}")
    lines.extend([
        "", "=== Runtime ===",
        f"recorded feature extraction time: {overall['feature_extraction_time_ms']:.6f} ms",
        "recorded feature extraction scope: all loaded frames; not used directly in speedup",
        f"training feature extraction time: {overall['training_feature_extraction_time_ms']:.6f} ms",
        f"decision feature extraction time: {overall['decision_feature_extraction_time_ms']:.6f} ms",
        f"training-decision overlap feature time: {overall['overlap_feature_extraction_time_ms']:.6f} ms",
        f"end-to-end unique feature extraction time: {overall['end_to_end_unique_feature_extraction_time_ms']:.6f} ms",
        f"total training time: {overall['total_training_time_ms']:.6f} ms",
        f"mean training time/window: {overall['mean_training_time_per_window_ms']:.6f} ms",
        f"total prediction time: {overall['total_prediction_time_ms']:.6f} ms",
        f"mean prediction time/window: {overall['mean_prediction_time_per_window_ms']:.6f} ms",
        f"prediction time/decision: {overall['prediction_time_per_decision_ms']:.6f} ms",
        f"prediction throughput: {overall['throughput_decisions_per_second']:.2f} decisions/s",
        f"effective application throughput: {overall['effective_application_throughput_frames_per_second']:.2f} application frames/s",
        f"total selection time: {overall['total_selection_time_ms']:.6f} ms",
        f"selection time/decision: {overall['selection_time_per_decision_ms']:.6f} ms",
    ])
    lines.extend([
        "", "=== Online Runtime Comparison ===",
        "note: trained-model runtime includes decision features, prediction, selection, and selected compression",
        f"decision feature frames: {runtime['decision_feature_frames']}",
        f"decision feature time: {runtime['decision_feature_time_ms']:.6f} ms",
        f"baseline trial rows: {runtime['baseline_trial_rows']}",
        f"baseline all trials time: {runtime['baseline_all_trials_time_ms']:.6f} ms",
        f"model prediction time: {runtime['model_prediction_time_ms']:.6f} ms",
        f"model selection time: {runtime['model_selection_time_ms']:.6f} ms",
        f"selected compression time: {runtime['selected_compression_time_ms']:.6f} ms",
        f"online total time: {runtime['online_total_time_ms']:.6f} ms",
        f"saved time: {runtime['saved_time_ms']:.6f} ms",
        f"time reduction: {runtime['time_reduction_pct']:.6f}%",
        f"speedup: {runtime['speedup']:.6f}x",
    ])
    lines.extend([
        "", "=== End-to-End Runtime ===",
        f"training feature frames: {runtime['training_feature_frames']}",
        f"training feature time: {runtime['training_feature_time_ms']:.6f} ms",
        f"training-decision overlap frames: {runtime['overlap_feature_frames']}",
        f"training-decision overlap feature time: {runtime['overlap_feature_time_ms']:.6f} ms",
        f"end-to-end unique feature frames: {runtime['end_to_end_unique_feature_frames']}",
        f"end-to-end unique feature time: {runtime['end_to_end_unique_feature_time_ms']:.6f} ms",
        f"training trial frames: {runtime['training_trial_frames']}",
        f"training trial rows: {runtime['training_trial_rows']}",
        f"training trial time: {runtime['training_trial_time_ms']:.6f} ms",
        f"model training time: {runtime['model_training_time_ms']:.6f} ms",
        f"model prediction time: {runtime['model_prediction_time_ms']:.6f} ms",
        f"model selection time: {runtime['model_selection_time_ms']:.6f} ms",
        f"non-trial selected compression time: {runtime['nontrial_selected_compression_time_ms']:.6f} ms",
        f"end-to-end baseline trial rows: {runtime['end_to_end_baseline_trial_rows']}",
        f"end-to-end baseline all trials time: {runtime['end_to_end_baseline_all_trials_time_ms']:.6f} ms",
        f"end-to-end online time: {runtime['end_to_end_online_time_ms']:.6f} ms",
        f"end-to-end saved time: {runtime['end_to_end_saved_time_ms']:.6f} ms",
        f"end-to-end time reduction: {runtime['end_to_end_time_reduction_pct']:.6f}%",
        f"end-to-end speedup: {runtime['end_to_end_speedup']:.6f}x",
    ])
    if saved:
        prediction_path, selection_path, timing_path, log_path = saved
        lines.extend(["", "=== Saved ===", f"prediction_csv: {prediction_path.resolve()}", f"selection_csv: {selection_path.resolve()}", f"compressor_timing_csv: {timing_path.resolve()}", f"summary_log: {log_path.resolve()}"])
    return "\n".join(lines)

def save_outputs(args: argparse.Namespace, result: dict[str, object]) -> tuple[Path, Path, Path, Path] | None:
    if not args.save_prefix:
        return None
    experiment_dir = Path(args.output_dir) / safe_experiment_name(args.save_prefix)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    active_metrics = result["active_metrics"]
    metadata = [
        "window_id", "model_id", "train_frame_start", "train_frame_end", "test_frame_start", "test_frame_end",
        "training_frame_count", "frame", "decision_frame", "application_frame", "tuning_granularity",
        "granularity_block_id", "block_offset", "is_decision_frame", "is_primary_prediction_frame",
        "is_training_window_frame", "is_selected_training_frame", "compressor", "eb",
    ]
    prediction_metrics = []
    for metric in active_metrics:
        actual_col, pred_col = METRIC_COLUMNS[metric]
        prediction_metrics.extend([actual_col, pred_col, f"{metric}_err", f"{metric}_rel_err", f"{metric}_accuracy"])
    prediction_columns = metadata + prediction_metrics + ["pred_joint_score", "actual_joint_score", "training_time_ms", "prediction_time_ms", "selection_time_ms"]
    if "compression_time_ms" in result["predictions"].columns:
        prediction_columns.append("compression_time_ms")
    prediction_path = experiment_dir / "joint_prediction_accuracy.csv"
    result["predictions"][prediction_columns].to_csv(prediction_path, index=False)
    selection_metadata = [
        "window_id", "model_id", "train_frame_start", "train_frame_end", "test_frame_start", "test_frame_end",
        "training_frame_count", "frame", "decision_frame", "application_frame", "tuning_granularity",
        "granularity_block_id", "block_offset", "is_decision_frame", "is_primary_prediction_frame",
        "is_training_window_frame", "is_selected_training_frame",
    ]
    decision_prediction_columns = [f"decision_{METRIC_COLUMNS[metric][1]}" for metric in active_metrics]
    selection_columns = selection_metadata + ["pred_compressor", "pred_eb"] + decision_prediction_columns + prediction_metrics
    selection_columns.extend(["decision_pred_joint_score", "pred_joint_score", "actual_joint_score_of_pred_choice", "oracle_compressor", "oracle_eb", "oracle_joint_score"])
    selection_columns.extend(f"oracle_{metric}" for metric in active_metrics)
    selection_columns.extend(["compressor_correct", "eb_correct", "pair_correct", "relative_score_gap_pct"])
    selection_columns.extend("psnr_gap_dB" if metric == "psnr" else f"{metric}_gap" for metric in active_metrics)
    if "compression_time_ms" in result["selection"].columns:
        selection_columns.append("compression_time_ms")
    selection_path = experiment_dir / "joint_selection_comparison.csv"
    result["selection"][selection_columns].to_csv(selection_path, index=False)
    timing_columns = [
        "window_id", "model_id", "target", "compressor", "train_frame_start", "train_frame_end",
        "test_frame_start", "test_frame_end", "training_frame_count", "decision_frames", "application_frames",
        "tuning_granularity", "train_rows", "test_rows", "training_time_ms", "prediction_time_ms", "skipped", "skip_reason",
    ]
    timing_path = experiment_dir / "compressor_timing.csv"
    result["compressor_timing"][timing_columns].to_csv(timing_path, index=False)
    log_path = experiment_dir / "run_summary.log"
    saved = (prediction_path, selection_path, timing_path, log_path)
    log_path.write_text(format_summary(args, result, saved) + "\n")
    return saved

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate online CR, PSNR-CR, SSIM-CR, or PSNR-SSIM-CR selection.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Offline result CSV containing the requested actual metrics.")
    parser.add_argument("--raw-data-path", default=DEFAULT_RAW_PATH, help="Original uint16 data file for Underwood features.")
    parser.add_argument("--raw-shape", type=parse_shape, default=(52224, 185, 194), help="Raw data shape: frames,height,width.")
    parser.add_argument("--train-window", type=int, default=16, help="Number of labeled training frames per window.")
    parser.add_argument("--test-window", type=int, default=16, help="Number of following frames predicted before retraining.")
    parser.add_argument("--window-mode", choices=["disjoint", "no_gap"], default="disjoint", help="disjoint preserves the original schedule; no_gap keeps the previous model active during the next training window.")
    parser.add_argument("--tuning-granularity", type=int, default=1, help="Number of consecutive application frames sharing one compressor/EB decision.")
    parser.add_argument("--train-frame-selection", choices=["all", "random", "odd", "even"], default="all", help="Training frames selected inside each window.")
    parser.add_argument("--selected-train-count", type=int, default=None, help="Number selected from random/odd/even candidates.")
    parser.add_argument("--random-train-count", type=int, default=None, help="Backward-compatible random selection count alias.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random training-frame selection seed.")
    parser.add_argument("--candidate-ebs", default=None, help="Comma-separated relative EB candidates. Default: all.")
    parser.add_argument("--candidate-compressors", default=None, help="Comma-separated compressors. Default: all.")
    parser.add_argument("--max-cr", type=float, default=None, help="Keep candidate rows with CR less than or equal to this value. Default: no upper limit.")
    parser.add_argument("--svd-energy", type=float, default=0.90, help="SVD energy retained by Underwood spatial feature.")
    parser.add_argument("--objective", choices=OBJECTIVE_METRICS, default="psnr_cr", help="Metrics predicted and combined for selection.")
    parser.add_argument("--psnr-weight", type=float, default=1.0, help="PSNR weight when PSNR is active.")
    parser.add_argument("--ssim-weight", type=float, default=1.0, help="SSIM weight when SSIM is active.")
    parser.add_argument("--cr-weight", type=float, default=1.0, help="CR weight; active weights are normalized automatically.")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Ridge regularization strength.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional first N candidate frames.")
    parser.add_argument("--quiet-features", action="store_true", help="Hide paper-feature progress output.")
    parser.add_argument("--save-prefix", default=None, help="Experiment folder name under output-dir.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory.")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    args.candidate_ebs = parse_float_list(args.candidate_ebs)
    args.candidate_compressors = parse_str_list(args.candidate_compressors)
    active_metrics = OBJECTIVE_METRICS[args.objective]
    normalized_active_weights(args, active_metrics)
    if args.tuning_granularity < 1:
        raise ValueError("tuning-granularity must be positive.")
    if not 0 < args.svd_energy <= 1:
        raise ValueError("svd-energy must be in (0, 1].")
    if args.max_cr is not None and args.max_cr <= 0:
        raise ValueError("max-cr must be greater than zero.")
    source_data = load_metric_csv(args.csv_path, replace_nonfinite=False)
    data = source_data.replace([np.inf, -np.inf], np.nan)
    required = ["frame", "compressor_id", "rel_bound", "cr", "frame_range", "data_variance", "compression_time_ms", "total_feature_time_ms"]
    if "psnr" in active_metrics:
        required.append("psnr")
    if "ssim" in active_metrics:
        required.extend(["ssim_pressio", "data_mean"])
    require_columns(data, required)
    application_data = filter_candidates(source_data, args)
    runtime_data = filter_candidates(data, args)
    runtime_data, cr_filter_summary = filter_max_cr(runtime_data, args.max_cr)
    data = prepare_base_features(runtime_data, active_metrics)
    if data.empty:
        raise ValueError("No usable candidate rows remain after filtering.")
    data, feature_time_ms, feature_source = add_features(data, args)
    result = run_window_evaluation(data, args, feature_time_ms, feature_source, runtime_data, application_data)
    result["cr_filter_summary"] = cr_filter_summary
    saved = save_outputs(args, result)
    print(format_summary(args, result, saved))

if __name__ == "__main__":
    main()
