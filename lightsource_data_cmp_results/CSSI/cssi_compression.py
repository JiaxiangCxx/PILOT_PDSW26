#!/usr/bin/env python3

from pathlib import Path
from time import perf_counter
import argparse
import csv
import fcntl
import h5py
import multiprocessing as mp
import numpy as np
import os
import libpressio

SCRIPT_DIR = Path(__file__).resolve()
PROJ_HOME = SCRIPT_DIR.parent

# change as needed 
FRAMES = 600
HEIGHT = 1813
WIDTH = 1558
MAX_FRAMES = FRAMES

GLOBAL_DATA = None
GLOBAL_FRAME_FEATURES = None
GLOBAL_EB_FEATURES = None

SVD_ENERGY = 0.90
EPS = 1e-12
DATA_SSIM_LEVELS = 256
DATA_SSIM_WINDOW = 11
DATA_SSIM_SIGMA = 1.5
DATA_SSIM_C1 = 1e-8
DATA_SSIM_C2 = 1e-8
CSV_COLUMNS = [
    "frame",
    "rel_bound",
    "abs_bound",
    "compressor_id",
    "cr",
    "mse",
    "psnr",
    "ssim_pressio",
    "data_ssim",
    "data_ssim_time_ms",
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

def fmt(x):
    if x is None:
        return "N/A"
    elif isinstance(x, float):
        return f"{x:.4g}"
    else:
        return str(x)

def load_uint32(path):
    data = np.fromfile(path, dtype=np.uint32)

    expected = FRAMES * HEIGHT * WIDTH
    if data.size != expected:
        raise ValueError(
            f"Size mismatch: file has {data.size} uint32 values, "
            f"but expected {expected} from shape "
            f"({FRAMES}, {HEIGHT}, {WIDTH})."
        )

    data = data.reshape((FRAMES, HEIGHT, WIDTH))
    print("data shape:", data.shape)
    print("data range:", data.min(), data.max())
    return data.astype(np.float32)


def load_fp32(path):
    data = np.fromfile(path, dtype=np.float32)

    expected = FRAMES * HEIGHT * WIDTH
    if data.size != expected:
        raise ValueError(
            f"Size mismatch: file has {data.size} float32 values, "
            f"but expected {expected} from shape "
            f"({FRAMES}, {HEIGHT}, {WIDTH})."
        )

    data = data.reshape((FRAMES, HEIGHT, WIDTH))
    print("data shape:", data.shape)
    print("data range:", data.min(), data.max())
    return data


def read_hdf5(path, data_field="data"):
    try:
        with h5py.File(path, "r") as hdf5_file:
            if data_field in hdf5_file:
                return hdf5_file[data_field][:]
            else:
                print(f"Dataset {data_field} not found in file {path}.")
                return None
    except Exception as e:
        print(f"An error occurred while reading the HDF5 file: {e}")
        return None

def gaussian_filter_valid(values, kernel):
    horizontal = np.tensordot(
        np.lib.stride_tricks.sliding_window_view(values, len(kernel), axis=1),
        kernel,
        axes=([-1], [0]),
    )
    return np.tensordot(
        np.lib.stride_tricks.sliding_window_view(horizontal, len(kernel), axis=0),
        kernel,
        axes=([-1], [0]),
    )

def calculate_data_ssim(reference, reconstructed):
    # Data SSIM follows Baker et al.: shared [0, 1] normalization, 256
    # quantization levels, and an 11 x 11 Gaussian window with sigma 1.5.
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Data SSIM requires two corresponding 2D arrays.")
    if min(x.shape) < DATA_SSIM_WINDOW:
        raise ValueError(
            f"Data SSIM requires both frame dimensions to be at least {DATA_SSIM_WINDOW}."
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan

    data_min = min(float(x.min()), float(y.min()))
    data_max = max(float(x.max()), float(y.max()))
    data_range = data_max - data_min
    if data_range <= 0:
        return 1.0

    scale = DATA_SSIM_LEVELS - 1
    x = np.rint(np.clip((x - data_min) / data_range, 0.0, 1.0) * scale) / scale
    y = np.rint(np.clip((y - data_min) / data_range, 0.0, 1.0) * scale) / scale

    radius = DATA_SSIM_WINDOW // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates ** 2) / (2.0 * DATA_SSIM_SIGMA ** 2))
    kernel /= kernel.sum()

    mean_x = gaussian_filter_valid(x, kernel)
    mean_y = gaussian_filter_valid(y, kernel)
    variance_x = gaussian_filter_valid(x * x, kernel) - mean_x * mean_x
    variance_y = gaussian_filter_valid(y * y, kernel) - mean_y * mean_y
    covariance = gaussian_filter_valid(x * y, kernel) - mean_x * mean_y
    numerator = (
        (2.0 * mean_x * mean_y + DATA_SSIM_C1)
        * (2.0 * covariance + DATA_SSIM_C2)
    )
    denominator = (
        (mean_x * mean_x + mean_y * mean_y + DATA_SSIM_C1)
        * (variance_x + variance_y + DATA_SSIM_C2)
    )
    local_scores = numerator / denominator
    return float(np.clip(np.mean(local_scores), -1.0, 1.0))

def quantized_entropy(frame, abs_bound):
    # Quantize the frame with the candidate absolute EB, then calculate Shannon entropy.
    if not np.isfinite(abs_bound) or abs_bound <= 0:
        return np.nan
    codes = np.floor((frame - frame.min()) / abs_bound).astype(np.int64)
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())

def svd_trunc_fraction(frame, energy=SVD_ENERGY):
    # A smaller retained fraction indicates stronger low-rank spatial structure.
    centered = frame.astype(np.float64) - float(np.mean(frame))
    scale = np.linalg.norm(centered)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    # SVD truncation is scale invariant; normalization avoids rejecting valid
    # scientific frames whose absolute values are much smaller than EPS.
    singular_values = np.linalg.svd(centered / scale, compute_uv=False)
    squared = singular_values ** 2
    total = squared.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    retained = int(np.searchsorted(np.cumsum(squared), energy * total) + 1)
    return retained / len(singular_values)

def precompute_cr_features(frame_indices, rel_bounds):
    # Compute frame-level features once per frame and entropy once per frame + EB.
    # The timer excludes raw-file loading, compression, decompression, and CSV writing.
    global GLOBAL_FRAME_FEATURES, GLOBAL_EB_FEATURES
    GLOBAL_FRAME_FEATURES = {}
    GLOBAL_EB_FEATURES = {}
    unique_rel_bounds = sorted({float(eb) for eb in rel_bounds})
    for frame_index in frame_indices:
        start = perf_counter()
        frame = GLOBAL_DATA[frame_index]
        data_min = float(np.min(frame))
        data_max = float(np.max(frame))
        frame_range = data_max - data_min
        data_variance = float(np.var(frame, dtype=np.float64))
        data_std = float(np.sqrt(max(data_variance, 0.0)))
        svd_fraction = svd_trunc_fraction(frame)
        entropy_by_eb = {}
        for rel_bound in unique_rel_bounds:
            abs_bound = float(rel_bound * frame_range)
            entropy_by_eb[rel_bound] = {
                "abs_bound": abs_bound,
                "quantized_entropy": quantized_entropy(frame, abs_bound),
            }
        total_feature_time_ms = (perf_counter() - start) * 1000.0
        GLOBAL_FRAME_FEATURES[frame_index] = {
            "frame_range": frame_range,
            "data_min": data_min,
            "data_max": data_max,
            "data_mean": float(np.mean(frame, dtype=np.float64)),
            "data_variance": data_variance,
            "data_std": data_std,
            "svd_trunc_fraction": svd_fraction,
            "total_feature_time_ms": total_feature_time_ms,
        }
        for rel_bound, values in entropy_by_eb.items():
            GLOBAL_EB_FEATURES[(frame_index, rel_bound)] = values

def run_compression_rel(compressor_id, rel_bound, frame_index):
    # if data_type == "uint32":
    #     full_data = load_uint32(input_file)
    #     data = full_data[frame_index]
    # elif data_type == "fp32":
    #     data = load_fp32(input_file)
    # elif data_type == "hdf5":
    #     data = read_hdf5(input_file)
    # else:
    #     raise ValueError(f"Unknown data_type: {data_type}")
    
    global GLOBAL_DATA, GLOBAL_FRAME_FEATURES, GLOBAL_EB_FEATURES

    # extract the single frame (2D)
    frame = np.ascontiguousarray(GLOBAL_DATA[frame_index], dtype=np.float32)
    frame_features = GLOBAL_FRAME_FEATURES[frame_index]
    eb_features = GLOBAL_EB_FEATURES[(frame_index, float(rel_bound))]

    # Apply the requested relative EB to this frame's own dynamic range.
    frame_range = frame_features["frame_range"]
    if frame_range <= 0:
        raise ValueError(f"Frame {frame_index} has zero data range.")
    abs_bound = eb_features["abs_bound"]

    # Validate the conversion internally; this redundant value is not saved to CSV.
    if not np.isclose(abs_bound / frame_range, rel_bound, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"Relative EB validation failed for frame {frame_index}.")

    # present the compressor with a 3D buffer (1, H, W) to match original behavior
    comp_input = np.ascontiguousarray(frame[np.newaxis, ...])
    recon_buffer = comp_input.copy()
    if compressor_id == "sz3":
        comp = libpressio.PressioCompressor.from_config({
            "compressor_id": "sz3",
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["size", "time", "error_stat", "ssim"],
            },
            "compressor_config": {
                "pressio:abs": float(abs_bound),
                "sz3:algorithm": 1,
                "sz3:l2_norm_error_bound": 0.333,
                "sz3:interp_algo": 1,
                "sz3:interp_direction": 0,
                "sz3:lorenzo": True,
                "sz3:lorenzo2": True,
                "sz3:regression": True,
                "sz3:regression2": True,
                "sz3:quant_bin_size": 1024,
            }
        })
    else:
        comp = libpressio.PressioCompressor.from_config({
            "compressor_id": compressor_id,
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["size", "time", "error_stat", "ssim"],
            },
            "compressor_config": {
                "pressio:abs": float(abs_bound),
            }
        })

    compression_start = perf_counter()
    comp_data = comp.encode(comp_input)
    compression_time_ms = (perf_counter() - compression_start) * 1000.0
    recon_buffer = comp.decode(comp_data, recon_buffer)

    # extract reconstructed 2D frame
    recon_frame = recon_buffer[0]

    metrics = comp.get_metrics()

    CR = metrics.get("size:compression_ratio")
    psnr = metrics.get("error_stat:psnr")
    ssim_pressio = metrics.get("ssim:ssim")
    data_ssim_start = perf_counter()
    data_ssim = calculate_data_ssim(frame, recon_frame)
    data_ssim_time_ms = (perf_counter() - data_ssim_start) * 1000.0
    comp_time = metrics.get("time:compress")
    decomp_time = metrics.get("time:decompress")
    recon_min = float(np.min(recon_frame))
    recon_max = float(np.max(recon_frame))
    recon_mean = float(np.mean(recon_frame, dtype=np.float64))
    recon_variance = float(np.var(recon_frame, dtype=np.float64))
    mse = float(np.mean((frame - recon_frame) ** 2, dtype=np.float64))

    print(
        f"{compressor_id}, rel={rel_bound:.1e}, "
        f"frame={frame_index}, CR={fmt(CR)}, "
        f"PSNR={fmt(psnr)}, SSIM={fmt(ssim_pressio)}, "
        f"DataSSIM={fmt(data_ssim)}, DataSSIMTimeMS={fmt(data_ssim_time_ms)}, "
        f"MSE={fmt(mse)}, "
        f"CompressionTimeMS={fmt(compression_time_ms)}, DecompTime={fmt(decomp_time)}"
    )

    return {
        "frame": frame_index,
        "rel_bound": rel_bound,
        "abs_bound": abs_bound,
        "compressor_id": compressor_id,
        "cr": CR,
        "mse": mse,
        "psnr": psnr,
        "ssim_pressio": ssim_pressio,
        "data_ssim": data_ssim,
        "data_ssim_time_ms": data_ssim_time_ms,
        "comp_time": comp_time,
        "compression_time_ms": compression_time_ms,
        "decomp_time": decomp_time,
        "frame_range": frame_range,
        "data_min": frame_features["data_min"],
        "data_max": frame_features["data_max"],
        "data_mean": frame_features["data_mean"],
        "data_variance": frame_features["data_variance"],
        "data_std": frame_features["data_std"],
        "svd_trunc_fraction": frame_features["svd_trunc_fraction"],
        "quantized_entropy": eb_features["quantized_entropy"],
        "total_feature_time_ms": frame_features["total_feature_time_ms"],
        "recon_min": recon_min,
        "recon_max": recon_max,
        "recon_mean": recon_mean,
        "recon_variance": recon_variance,
    }


def mp_worker(args):
    return run_compression_rel(*args)


def job_key(frame_index, compressor_id, rel_bound):
    return int(frame_index), str(compressor_id), float(rel_bound)


def load_completed_jobs(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"Cannot resume {path}: CSV columns do not match this full-frame experiment."
            )
        completed = set()
        for row in reader:
            try:
                completed.add(
                    job_key(row["frame"], row["compressor_id"], row["rel_bound"])
                )
            except (TypeError, ValueError):
                continue
    return completed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compress complete CSSI frames and calculate full-frame Data SSIM."
    )
    parser.add_argument("--input-file", default="/home/jzhang84/lsCOMP-AD-AE/cssi-600.bin")
    parser.add_argument("--input-dtype", choices=["uint32", "float32"], default="uint32")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel compression workers. Full-frame Data SSIM is memory intensive.",
    )
    parser.add_argument("--output-csv", default=None)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def main():
    global FRAMES, HEIGHT, WIDTH, MAX_FRAMES, GLOBAL_DATA
    args = build_parser().parse_args()
    FRAMES = args.frames
    HEIGHT = args.height
    WIDTH = args.width
    MAX_FRAMES = args.max_frames
    if min(FRAMES, HEIGHT, WIDTH, MAX_FRAMES, args.workers) <= 0:
        raise ValueError("Frames, dimensions, max-frames, and workers must be positive.")

    # input_file = "/home/jzhang84/lsCOMP-AD-AE/cssi-128.bin" #  128 1813 1558
    input_file = args.input_file # 600 1813 1558
    requested_frames = min(MAX_FRAMES, FRAMES)
    out_csv = Path(args.output_csv) if args.output_csv else (
        PROJ_HOME
        / f"cssi_full_frame_results_{requested_frames}_relEB-2-6-10_data_ssim.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{out_csv}.lock")
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(
            f"Another compression run is already using {out_csv}."
        ) from exc
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()

    # CSSI detector values are stored as uint32; convert the decoded values
    # to float32 so every compressor receives the same numeric input type.
    if args.input_dtype == "uint32":
        GLOBAL_DATA = load_uint32(input_file)
    else:
        GLOBAL_DATA = load_fp32(input_file)
    # filter out zero-value frames (ptp == 0) to avoid trivial ssim=1 frames
    total_frames = GLOBAL_DATA.shape[0]
    zero_frames = [i for i in range(total_frames) if np.ptp(GLOBAL_DATA[i]) == 0]
    valid_frames = [i for i in range(total_frames) if np.ptp(GLOBAL_DATA[i]) != 0]
    if not valid_frames:
        raise ValueError("All frames have zero dynamic range; relative EB cannot be applied.")
    frames_to_process = valid_frames[:min(MAX_FRAMES, len(valid_frames))]
    print(f"Filtered out {len(zero_frames)} zero-ptp frames; {len(valid_frames)} valid frames remain.")
    print(f"MAX_FRAMES={MAX_FRAMES}; processing {len(frames_to_process)} frames.")

    rel_bounds_by_comp = {
           "sz": np.logspace(-2, -6, 10),
           "sz3": np.logspace(-2, -6, 10),
           "sperr": np.logspace(-2, -6, 10),
           "zfp": np.logspace(-2, -6, 10),
           "szx": np.logspace(-2, -6, 10),
           "mgard": np.logspace(-2, -6, 10),
    }
    compressors = list(rel_bounds_by_comp.keys())
    jobs = [
        (comp, eb, frame_index)
        for comp in compressors
        for eb in rel_bounds_by_comp[comp]
        for frame_index in frames_to_process
    ]

    if out_csv.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{out_csv} already exists. Use --resume to continue or --overwrite to replace it."
        )

    completed = load_completed_jobs(out_csv) if args.resume else set()
    pending_jobs = [
        job for job in jobs
        if job_key(job[2], job[0], job[1]) not in completed
    ]
    print(
        f"Expected trials={len(jobs)}; completed={len(completed)}; "
        f"remaining={len(pending_jobs)}."
    )
    if not pending_jobs:
        print("All requested trials are already complete.")
        print("Saved:", out_csv)
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()
        return

    # Precompute model inputs before launching compression workers.
    pending_frames = sorted({job[2] for job in pending_jobs})
    pending_rel_bounds = sorted({float(job[1]) for job in pending_jobs})
    precompute_cr_features(pending_frames, pending_rel_bounds)

    append = args.resume and out_csv.exists() and out_csv.stat().st_size > 0
    with out_csv.open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        if not append:
            writer.writeheader()
            stream.flush()
        completed_this_run = 0
        with mp.Pool(processes=min(len(pending_jobs), args.workers)) as pool:
            for result in pool.imap_unordered(mp_worker, pending_jobs, chunksize=1):
                writer.writerow(result)
                stream.flush()
                completed_this_run += 1
                print(
                    f"progress: {len(completed) + completed_this_run}/{len(jobs)} "
                    f"trials written"
                )

    print(f"Completed {completed_this_run} new trials.")
    print("Saved:", out_csv)
    fcntl.flock(lock_stream, fcntl.LOCK_UN)
    lock_stream.close()


if __name__ == "__main__":
    main()
