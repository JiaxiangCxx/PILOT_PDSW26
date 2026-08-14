#!/usr/bin/env python3
"""Frame-relative compression experiment for the APS REI dataset.

Adjustable experiment variables:
    DATA_PATH: input GE5 scan.
    MAX_FRAMES: maximum number of non-constant frames to process.
    REL_BOUNDS: shared candidate relative error bounds.
    REL_BOUNDS_BY_COMP: compressors and their candidate error bounds.
    PROCESSES: parallel compression workers.
    OUTPUT_CSV: result CSV path.

This script only performs compression-quality experiments. It does not train
the REI model, calculate REI scores, or write reconstructed GE5/chunk files.
"""

from pathlib import Path
from time import perf_counter
import csv
import multiprocessing as mp
import os
import numpy as np
import libpressio

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/home/jzhang84/lsCOMP-AD-AE/APS_data/test.edf.ge5")
OUTPUT_CSV = SCRIPT_DIR / "rei_results_1440_relEB-2-6-10_rerun.csv"

FRAMES = 1440
HEIGHT = 2048
WIDTH = 2048
HEADER_BYTES = 8396800
RAW_DTYPE = np.dtype(np.uint16)
COMPRESSION_DTYPE = np.dtype(np.float32)
MAX_FRAMES = FRAMES
PROCESSES = 11
SVD_ENERGY = 0.90
EPS = 1e-12

REL_BOUNDS = np.logspace(-2, -6, 10)
REL_BOUNDS_BY_COMP = {
    "sz": REL_BOUNDS,
    "sz3": REL_BOUNDS,
    "sperr": REL_BOUNDS,
    "zfp": REL_BOUNDS,
    "szx": REL_BOUNDS,
    "mgard": REL_BOUNDS,
}

CSV_COLUMNS = [
    "frame",
    "rel_bound",
    "abs_bound",
    "compressor_id",
    "cr",
    "mse",
    "psnr",
    "ssim_pressio",
    "comp_time",
    "compression_time_ms",
    "decomp_time",
    "frame_range",
    "data_min",
    "data_max",
    "data_mean",
    "data_variance",
    "data_std",
    "svd_trunc_fraction",
    "quantized_entropy",
    "total_feature_time_ms",
    "recon_min",
    "recon_max",
    "recon_mean",
    "recon_variance",
]

GLOBAL_DATA = None


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def open_ge5_memmap(path):
    path = Path(path)
    expected_payload = FRAMES * HEIGHT * WIDTH * RAW_DTYPE.itemsize
    expected_size = HEADER_BYTES + expected_payload
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"GE5 size mismatch: {path} has {actual_size} bytes, "
            f"expected {expected_size} bytes for header={HEADER_BYTES}, "
            f"shape=({FRAMES}, {HEIGHT}, {WIDTH}), dtype={RAW_DTYPE}."
        )
    return np.memmap(
        path,
        dtype=RAW_DTYPE,
        mode="r",
        offset=HEADER_BYTES,
        shape=(FRAMES, HEIGHT, WIDTH),
        order="C",
    )


def init_worker(data_path):
    global GLOBAL_DATA
    GLOBAL_DATA = open_ge5_memmap(data_path)


def find_valid_frames(data, max_frames):
    valid_frames = []
    constant_frames = []
    requested = min(max_frames, FRAMES)
    print(
        f"[SCAN] finding up to {requested} non-constant frames "
        f"from {FRAMES} total frames",
        flush=True,
    )
    for frame_index in range(FRAMES):
        frame = data[frame_index]
        if np.ptp(frame) == 0:
            constant_frames.append(frame_index)
        else:
            valid_frames.append(frame_index)
        if (frame_index + 1) % 25 == 0 or len(valid_frames) == requested:
            print(
                f"[SCAN] checked={frame_index + 1}/{FRAMES} | "
                f"valid={len(valid_frames)}/{requested} | "
                f"constant={len(constant_frames)}",
                flush=True,
            )
        if len(valid_frames) == requested:
            break
    if not valid_frames:
        raise ValueError("All scanned frames have zero data range.")
    return valid_frames, constant_frames


def full_svd_trunc_fraction(frame, energy=SVD_ENERGY):
    # Full 2048x2048 SVD is calculated exactly once for each selected frame.
    centered = frame.astype(np.float64) - float(np.mean(frame, dtype=np.float64))
    scale = np.linalg.norm(centered)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    singular_values = np.linalg.svd(centered / scale, compute_uv=False)
    squared = singular_values**2
    total = float(squared.sum())
    if not np.isfinite(total) or total <= 0:
        return np.nan
    retained = int(np.searchsorted(np.cumsum(squared), energy * total) + 1)
    return retained / len(singular_values)


def quantized_entropy(frame, data_min, abs_bound):
    if not np.isfinite(abs_bound) or abs_bound <= 0:
        return np.nan
    codes = np.floor((frame - data_min) / abs_bound).astype(np.int64)
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_frame_features(data, frame_index, rel_bounds):
    start = perf_counter()
    frame = np.asarray(data[frame_index], dtype=COMPRESSION_DTYPE)
    data_min = float(np.min(frame))
    data_max = float(np.max(frame))
    frame_range = data_max - data_min
    if frame_range <= 0:
        raise ValueError(f"Frame {frame_index} has zero data range.")
    data_mean = float(np.mean(frame, dtype=np.float64))
    data_variance = float(np.var(frame, dtype=np.float64))
    data_std = float(np.sqrt(max(data_variance, 0.0)))
    svd_fraction = full_svd_trunc_fraction(frame)
    eb_features = {}
    for rel_bound in sorted({float(value) for value in rel_bounds}, reverse=True):
        abs_bound = float(rel_bound * frame_range)
        eb_features[rel_bound] = {
            "abs_bound": abs_bound,
            "quantized_entropy": quantized_entropy(frame, data_min, abs_bound),
        }
    total_feature_time_ms = (perf_counter() - start) * 1000.0
    frame_features = {
        "frame_range": frame_range,
        "data_min": data_min,
        "data_max": data_max,
        "data_mean": data_mean,
        "data_variance": data_variance,
        "data_std": data_std,
        "svd_trunc_fraction": svd_fraction,
        "total_feature_time_ms": total_feature_time_ms,
    }
    return frame_features, eb_features


def make_compressor(compressor_id, abs_bound):
    compressor_config = {"pressio:abs": float(abs_bound)}
    if compressor_id == "sz3":
        compressor_config.update(
            {
                "sz3:algorithm": 1,
                "sz3:l2_norm_error_bound": 0.333,
                "sz3:interp_algo": 1,
                "sz3:interp_direction": 0,
                "sz3:lorenzo": True,
                "sz3:lorenzo2": True,
                "sz3:regression": True,
                "sz3:regression2": True,
            }
        )
    return libpressio.PressioCompressor.from_config(
        {
            "compressor_id": compressor_id,
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["size", "time", "error_stat", "ssim"],
            },
            "compressor_config": compressor_config,
        }
    )


def compress_one(args):
    compressor_id, rel_bound, frame_index, frame_features, eb_features = args
    frame = np.asarray(GLOBAL_DATA[frame_index], dtype=COMPRESSION_DTYPE)
    recon_frame = np.empty_like(frame)
    abs_bound = eb_features["abs_bound"]
    frame_range = frame_features["frame_range"]
    if not np.isclose(abs_bound / frame_range, rel_bound, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"Relative EB validation failed for frame {frame_index}.")
    compressor = make_compressor(compressor_id, abs_bound)
    compression_start = perf_counter()
    comp_data = compressor.encode(frame)
    compression_time_ms = (perf_counter() - compression_start) * 1000.0
    recon_frame = compressor.decode(comp_data, recon_frame)
    metrics = compressor.get_metrics()
    difference = frame - recon_frame
    result = {
        "frame": frame_index,
        "rel_bound": rel_bound,
        "abs_bound": abs_bound,
        "compressor_id": compressor_id,
        "cr": metrics.get("size:compression_ratio"),
        "mse": float(np.mean(difference * difference, dtype=np.float64)),
        "psnr": metrics.get("error_stat:psnr"),
        "ssim_pressio": metrics.get("ssim:ssim"),
        "comp_time": metrics.get("time:compress"),
        "compression_time_ms": compression_time_ms,
        "decomp_time": metrics.get("time:decompress"),
        **frame_features,
        "quantized_entropy": eb_features["quantized_entropy"],
        "recon_min": float(np.min(recon_frame)),
        "recon_max": float(np.max(recon_frame)),
        "recon_mean": float(np.mean(recon_frame, dtype=np.float64)),
        "recon_variance": float(np.var(recon_frame, dtype=np.float64)),
    }
    return {"ok": True, "result": result}


def compression_worker(args):
    compressor_id, rel_bound, frame_index, _, _ = args
    try:
        return compress_one(args)
    except Exception as exc:
        return {
            "ok": False,
            "frame": frame_index,
            "compressor_id": compressor_id,
            "rel_bound": rel_bound,
            "error": f"{type(exc).__name__}: {exc}",
        }


def print_result(result, completed, total_jobs):
    print(
        f"[COMPRESS {completed}/{total_jobs}] "
        f"frame={result['frame']} | "
        f"compressor={result['compressor_id']} | "
        f"relEB={result['rel_bound']:.6g} | "
        f"absEB={result['abs_bound']:.6g} | "
        f"CR={fmt(result['cr'])} | "
        f"PSNR={fmt(result['psnr'])} | "
        f"SSIM={fmt(result['ssim_pressio'])} | "
        f"compression_time_ms={result['compression_time_ms']:.6f}",
        flush=True,
    )


def main():
    global GLOBAL_DATA
    print("=== REI Compression Experiment ===", flush=True)
    print(f"input: {DATA_PATH}", flush=True)
    print(f"shape: ({FRAMES}, {HEIGHT}, {WIDTH})", flush=True)
    print(f"raw dtype: {RAW_DTYPE}; compression dtype: {COMPRESSION_DTYPE}", flush=True)
    print(f"MAX_FRAMES: {MAX_FRAMES}", flush=True)
    print(f"compressors: {list(REL_BOUNDS_BY_COMP)}", flush=True)
    print(f"relative EBs: {[float(value) for value in REL_BOUNDS]}", flush=True)
    GLOBAL_DATA = open_ge5_memmap(DATA_PATH)
    frames_to_process, constant_frames = find_valid_frames(GLOBAL_DATA, MAX_FRAMES)
    compressors = list(REL_BOUNDS_BY_COMP)
    all_rel_bounds = [
        value
        for compressor in compressors
        for value in REL_BOUNDS_BY_COMP[compressor]
    ]
    unique_rel_bounds = sorted({float(value) for value in all_rel_bounds}, reverse=True)
    jobs_per_frame = sum(len(REL_BOUNDS_BY_COMP[compressor]) for compressor in compressors)
    total_jobs = len(frames_to_process) * jobs_per_frame
    print(
        f"[SETUP] selected_frames={len(frames_to_process)} | "
        f"constant_frames={len(constant_frames)} | "
        f"compressors={len(compressors)} | "
        f"error_bounds={len(unique_rel_bounds)} | "
        f"total_jobs={total_jobs}",
        flush=True,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    successful = 0
    failed = 0
    experiment_start = perf_counter()
    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_file.flush()
        with mp.Pool(
            processes=min(PROCESSES, jobs_per_frame, os.cpu_count() or PROCESSES),
            initializer=init_worker,
            initargs=(str(DATA_PATH),),
        ) as pool:
            for frame_position, frame_index in enumerate(frames_to_process, start=1):
                print(
                    f"[FRAME {frame_position}/{len(frames_to_process)}] "
                    f"frame={frame_index} feature extraction started",
                    flush=True,
                )
                frame_features, eb_features_by_rel = extract_frame_features(
                    GLOBAL_DATA,
                    frame_index,
                    unique_rel_bounds,
                )
                print(
                    f"[FRAME {frame_position}/{len(frames_to_process)}] "
                    f"frame={frame_index} feature extraction completed | "
                    f"range={frame_features['frame_range']:.6g} | "
                    f"svd_trunc_fraction={frame_features['svd_trunc_fraction']:.6g} | "
                    f"feature_time_ms={frame_features['total_feature_time_ms']:.6f}",
                    flush=True,
                )
                jobs = [
                    (
                        compressor,
                        float(rel_bound),
                        frame_index,
                        frame_features,
                        eb_features_by_rel[float(rel_bound)],
                    )
                    for compressor in compressors
                    for rel_bound in REL_BOUNDS_BY_COMP[compressor]
                ]
                for outcome in pool.imap_unordered(compression_worker, jobs, chunksize=1):
                    completed += 1
                    if outcome["ok"]:
                        result = outcome["result"]
                        writer.writerow(result)
                        csv_file.flush()
                        successful += 1
                        print_result(result, completed, total_jobs)
                    else:
                        failed += 1
                        print(
                            f"[ERROR {completed}/{total_jobs}] "
                            f"frame={outcome['frame']} | "
                            f"compressor={outcome['compressor_id']} | "
                            f"relEB={outcome['rel_bound']:.6g} | "
                            f"{outcome['error']}",
                            flush=True,
                        )
    elapsed = perf_counter() - experiment_start
    print("=== Completed ===", flush=True)
    print(f"successful_jobs: {successful}", flush=True)
    print(f"failed_jobs: {failed}", flush=True)
    print(f"elapsed_seconds: {elapsed:.6f}", flush=True)
    print(f"saved: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
