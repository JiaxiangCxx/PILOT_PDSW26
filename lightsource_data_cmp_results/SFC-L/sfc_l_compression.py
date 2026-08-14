#!/usr/bin/env python3

from pathlib import Path
from time import perf_counter
import csv
import h5py
import multiprocessing as mp
import numpy as np
import libpressio

SCRIPT_DIR = Path(__file__).resolve()
PROJ_HOME = SCRIPT_DIR.parent

# change as needed 
FRAMES = 52224 
HEIGHT = 185      
WIDTH = 194  

GLOBAL_DATA = None
GLOBAL_RANGE = None
GLOBAL_FRAME_FEATURES = None
GLOBAL_EB_FEATURES = None

SVD_ENERGY = 0.90
EPS = 1e-12

# CR model features are computed from the original frame before compression.
# They can therefore be used to predict CR for every candidate compressor + EB pair.

WINDOW_SIZE = 10      # frames used for estimation
UPDATE_INTERVAL = 110 # 10 warmup + 100 prediction frames

def get_adaptive_range(frame_index):

    global GLOBAL_DATA

    block_start = (frame_index // UPDATE_INTERVAL) * UPDATE_INTERVAL

    warmup_start = block_start

    warmup_end = min(warmup_start + WINDOW_SIZE, GLOBAL_DATA.shape[0])

    warmup_data = GLOBAL_DATA[warmup_start:warmup_end]

    return float(np.ptp(warmup_data))

def fmt(x):
    if x is None:
        return "N/A"
    elif isinstance(x, float):
        return f"{x:.4g}"
    else:
        return str(x)

def load_uint16(path):
    data = np.fromfile(path, dtype=np.uint16)

    expected = FRAMES * HEIGHT * WIDTH
    if data.size != expected:
        raise ValueError(
            f"Size mismatch: file has {data.size} uint16 values, "
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

    return data.reshape((FRAMES, HEIGHT, WIDTH))


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

def quantized_entropy(frame, abs_bound):
    # Quantize the frame with the candidate absolute EB, then calculate Shannon entropy.
    # Lower entropy generally indicates that the quantized values are easier to compress.

    # Original frame values -> Divide values into bins using abs_bound -> Count how many values fall in each bin -> Convert counts to probabilities -> Calculate Shannon entropy
    if not np.isfinite(abs_bound) or abs_bound <= 0:
        return np.nan
    codes = np.floor((frame - frame.min()) / abs_bound).astype(np.int64)
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())

def svd_trunc_fraction(frame, energy=SVD_ENERGY):
    # Singular Value Decomposition
    # A smaller fraction indicates stronger spatial redundancy in the frame.
    centered = frame.astype(np.float64) - float(np.mean(frame))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    squared = singular_values ** 2
    total = squared.sum()
    if total <= EPS:
        return np.nan
    retained = int(np.searchsorted(np.cumsum(squared), energy * total) + 1)
    return retained / len(singular_values)

def precompute_cr_features(frame_indices, rel_bounds):
    # Compute frame statistics/SVD once per frame and entropy once per frame + EB.
    # total_feature_time_ms is the PSNR+CR input-feature cost for all candidate EBs of one frame.
    # The total includes min/max/range, variance/std, SVD, and all entropy values.
    # It excludes compression, decompression, CSV writing, and raw-file loading time.
    # data_mean is retained in the CSV for analysis but is calculated outside this timer.
    global GLOBAL_FRAME_FEATURES, GLOBAL_EB_FEATURES
    GLOBAL_FRAME_FEATURES = {}
    GLOBAL_EB_FEATURES = {}
    unique_rel_bounds = sorted({float(eb) for eb in rel_bounds})

    for position, frame_index in enumerate(frame_indices, start=1):
        # Start one timer per frame so online feature overhead is not counted per compressor.
        start = perf_counter()
        data = GLOBAL_DATA[frame_index]
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        frame_range = data_max - data_min
        data_variance = float(np.var(data, dtype=np.float64))
        data_std = float(np.sqrt(max(data_variance, 0.0)))
        svd_fraction = svd_trunc_fraction(data)

        entropy_by_eb = {}
        for rel_bound in unique_rel_bounds:
            abs_bound = float(rel_bound * frame_range)
            entropy_by_eb[rel_bound] = {
                "abs_bound": abs_bound,
                "quantized_entropy": quantized_entropy(data, abs_bound),
            }

        # This value is repeated in every CSV row for the frame; deduplicate by frame before summing.
        total_feature_time_ms = (perf_counter() - start) * 1000.0
        data_mean = float(np.mean(data, dtype=np.float64))
        GLOBAL_FRAME_FEATURES[frame_index] = {
            "frame_range": frame_range,
            "data_min": data_min,
            "data_max": data_max,
            "data_mean": data_mean,
            "data_variance": data_variance,
            "data_std": data_std,
            "svd_trunc_fraction": svd_fraction,
            "total_feature_time_ms": total_feature_time_ms,
        }
        for rel_bound, values in entropy_by_eb.items():
            GLOBAL_EB_FEATURES[(frame_index, rel_bound)] = values

        # if position % 100 == 0 or position == len(frame_indices):
        #     print(f"CR features: {position}/{len(frame_indices)} frames")

def run_compression_rel(compressor_id, rel_bound, frame_index, stream_index):
    # if data_type == "uint16":
    #     full_data = load_uint16(input_file)
    #     data = full_data[frame_index]
    # elif data_type == "fp32":
    #     data = load_fp32(input_file)
    # elif data_type == "hdf5":
    #     data = read_hdf5(input_file)
    # else:
    #     raise ValueError(f"Unknown data_type: {data_type}")
    
    global GLOBAL_DATA, GLOBAL_RANGE, GLOBAL_FRAME_FEATURES, GLOBAL_EB_FEATURES

    data=GLOBAL_DATA[frame_index] 
    recon_data=data.copy()
    frame_features = GLOBAL_FRAME_FEATURES[frame_index]
    eb_features = GLOBAL_EB_FEATURES[(frame_index, float(rel_bound))]

    # Apply the same relative error bound to every frame using its own data range.
    # The absolute bound therefore changes with each frame's dynamic range.
    frame_range = frame_features["frame_range"]
    if frame_range <= 0:
        raise ValueError(f"Frame {frame_index} has zero data range.")
    abs_bound = eb_features["abs_bound"]

    # Internal validation only; effective_rel_eb is intentionally not saved to CSV.
    if not np.isclose(abs_bound / frame_range, rel_bound, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"Relative EB validation failed for frame {frame_index}.")

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

    # Wall-clock compression time in milliseconds for this compressor + EB pair.
    # comp_time below is retained separately because it comes from libpressio's time metric.
    compression_start = perf_counter()
    comp_data=comp.encode(data)
    compression_time_ms = (perf_counter() - compression_start) * 1000.0
    recon_data = comp.decode(comp_data, recon_data)

    metrics = comp.get_metrics()

    CR = metrics.get("size:compression_ratio")
    psnr = metrics.get("error_stat:psnr")
    ssim_pressio = metrics.get("ssim:ssim")
    comp_time = metrics.get("time:compress")
    decomp_time = metrics.get("time:decompress")

    recon_min=recon_data.min()
    recon_max=recon_data.max()
    recon_mean=float(np.mean(recon_data, dtype=np.float64))
    recon_variance=float(np.var(recon_data, dtype=np.float64))

    mse=np.mean((data-recon_data)**2)

    print(
        f"{compressor_id}, rel={rel_bound:.1e}, "
        f"frame={frame_index}, CR={fmt(CR)}, "
        f"PSNR={fmt(psnr)}, SSIM={fmt(ssim_pressio)}, MSE={fmt(mse)}, "
        f"CompressionTimeMS={fmt(compression_time_ms)}, DecompTime={fmt(decomp_time)}"
    )

    return {
        "frame_index": frame_index,
        "rel_bound": rel_bound,
        "abs_bound": abs_bound,
        "compressor_id": compressor_id,
        "cr": CR,
        "psnr": psnr,
        "ssim_pressio": ssim_pressio,
        "mse": mse,
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


if __name__ == "__main__":

    input_file = "/home/jzhang84/lsCOMP-AD-AE/0-sfc.uint16" # 52224 185 194 

    GLOBAL_DATA = load_uint16(input_file)
    GLOBAL_RANGE = float(np.ptp(GLOBAL_DATA))

    # filter out zero-range frames where min == max.
    # These frames have no dynamic range and can give trivial metrics.
    total_frames = GLOBAL_DATA.shape[0]

    valid_frames = []
    zero_range_frames = []

    for i in range(total_frames):
        
        frame = GLOBAL_DATA[i]

        if np.ptp(frame) == 0:
            zero_range_frames.append(i)
        else:
            valid_frames.append(i)

    print("removed zero-range frames:", len(zero_range_frames))
    print("remaining valid frames:", len(valid_frames))

    FRAME_INDEX = np.arange(min(10000, len(valid_frames)), dtype=int)

    # FRAME_INDEX = np.linspace(0, len(valid_frames) - 1, 100, dtype=int)
    # FRAME_INDEX = np.linspace(0, 10, 10, dtype=int)
    FRAME_INDEX = [valid_frames[i] for i in FRAME_INDEX]

    # compressors = ["sz3", "sz", "zfp", "mgard", "szx"，"sperr"]
    # rel_bounds = [1e-3]

    rel_bounds_by_comp = {
        "sz": np.logspace(-2, -6, 10),
        "sz3": np.logspace(-2, -6, 10),
        "sperr": np.logspace(-2, -6, 10),
        "zfp": np.logspace(-2, -6, 10),
        "szx": np.logspace(-2, -6, 10),
        "mgard": np.logspace(-2, -6, 10),
    }
    compressors = list(rel_bounds_by_comp.keys())    

    candidate_rel_bounds = [
        eb
        for bounds in rel_bounds_by_comp.values()
        for eb in bounds
    ]
    # Precompute model inputs before launching compression workers to avoid repeating SVD/entropy.
    precompute_cr_features(FRAME_INDEX, candidate_rel_bounds)

    jobs = [
        (comp, eb, frame_index, i)
        for comp in compressors
        for eb in rel_bounds_by_comp[comp]
        for i, frame_index in enumerate(FRAME_INDEX)
    ]

    out_csv = (
        PROJ_HOME
        / "sfc-l_results_10000_relEB-2-6-10.csv"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Feature columns are saved with the quality metrics for direct CR-model training.
        writer.writerow([
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
        ])

        f.flush()

        with mp.Pool(processes=min(len(jobs), 11)) as pool:
            for r in pool.imap_unordered(mp_worker, jobs, chunksize=1):
                writer.writerow([
                    r["frame_index"],
                    r["rel_bound"],
                    r["abs_bound"],
                    r["compressor_id"],
                    r["cr"],
                    r["mse"],
                    r["psnr"],
                    r["ssim_pressio"],
                    r["comp_time"],
                    r["compression_time_ms"],
                    r["decomp_time"],
                    r["frame_range"],
                    r["data_min"],
                    r["data_max"],
                    r["data_mean"],
                    r["data_variance"],
                    r["data_std"],
                    r["svd_trunc_fraction"],
                    r["quantized_entropy"],
                    r["total_feature_time_ms"],
                    r["recon_min"],
                    r["recon_max"],
                    r["recon_mean"],
                    r["recon_variance"],
                ])
                f.flush()

    print("Saved:", out_csv)
