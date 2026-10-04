"""Comprehensive benchmark: PyTorch vs ONNX, CPU vs GPU.

Measures per-model:
  - File size
  - Average / p95 latency (ms)
  - FPS
  - RAM delta
  - GPU memory peak (if applicable)

Outputs a comparison table suitable for a technical report.
"""

from __future__ import annotations

import os

# Disable Ultralytics auto-install (prevents onnxruntime CPU being installed
# over onnxruntime-gpu, which would break GPU inference).
os.environ["YOLO_AUTOINSTALL"] = "false"

import gc
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from ultralytics import YOLO


# ============================================================
# Config
# ============================================================

PT_MODEL = "models/pytorch/yolov8n.pt"
ONNX_MODEL = "models/onnx/yolov8n.onnx"

WARMUP = 10
RUNS = 50
IMGSZ = 640
INPUT_SHAPE = (720, 1280, 3)


# ============================================================
# Helpers
# ============================================================

def _human_mb(bytes_val: float) -> float:
    return round(bytes_val / 1024 / 1024, 2)


def _file_size_mb(path: str) -> float:
    p = Path(path)
    return _human_mb(p.stat().st_size) if p.exists() else 0.0


def _process_ram_mb() -> float:
    return _human_mb(psutil.Process().memory_info().rss)


def _make_frame() -> np.ndarray:
    return np.random.randint(0, 255, INPUT_SHAPE, dtype=np.uint8)


# ============================================================
# Benchmark core
# ============================================================

def benchmark(
    model_path: str,
    device: str,
    imgsz: int = IMGSZ,
    warmup: int = WARMUP,
    runs: int = RUNS,
) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(model_path, task="detect")
    frame = _make_frame()

    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)

    ram_before = _process_ram_mb()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        times.append(time.perf_counter() - t0)
    ram_after = _process_ram_mb()

    avg = statistics.mean(times)
    p95 = float(np.percentile(times, 95))

    gpu_mem_mb = 0.0
    if device != "cpu" and torch.cuda.is_available():
        gpu_mem_mb = _human_mb(torch.cuda.max_memory_allocated())

    return {
        "model_path": model_path,
        "device": device,
        "avg_ms": round(avg * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "fps": round(1.0 / avg, 1),
        "size_mb": _file_size_mb(model_path),
        "ram_delta_mb": round(ram_after - ram_before, 2),
        "gpu_mem_mb": round(gpu_mem_mb, 2),
    }


# ============================================================
# Reporting
# ============================================================

def print_table(results: list[dict]) -> None:
    headers = ["Modèle", "Device", "Taille (MB)", "Latence avg",
               "Latence p95", "FPS", "RAM Δ (MB)", "GPU (MB)"]
    widths = [22, 8, 12, 12, 12, 8, 12, 10]

    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + "=" * len(line))
    print(line)
    print("=" * len(line))

    for r in results:
        name = Path(r["model_path"]).name
        dev = "GPU" if r["device"] != "cpu" else "CPU"
        row = [
            name,
            dev,
            f"{r['size_mb']:.2f}",
            f"{r['avg_ms']:.2f} ms",
            f"{r['p95_ms']:.2f} ms",
            f"{r['fps']:.1f}",
            f"{r['ram_delta_mb']:.1f}",
            f"{r['gpu_mem_mb']:.1f}" if r["gpu_mem_mb"] > 0 else "—",
        ]
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))

    print("=" * len(line) + "\n")


def print_speedups(results: list[dict]) -> None:
    by_key = {(Path(r["model_path"]).suffix, r["device"]): r for r in results}

    print("📊 Speedups :")
    print("-" * 60)

    for device in ("cpu", "0"):
        pt = by_key.get((".pt", device))
        onnx = by_key.get((".onnx", device))
        if pt and onnx:
            dev_name = "GPU" if device != "cpu" else "CPU"
            speedup = pt["avg_ms"] / onnx["avg_ms"] if onnx["avg_ms"] else 0
            print(f"  ONNX vs PyTorch ({dev_name})   : x{speedup:.2f}")

    for suffix, name in ((".pt", "PyTorch"), (".onnx", "ONNX")):
        cpu = by_key.get((suffix, "cpu"))
        gpu = by_key.get((suffix, "0"))
        if cpu and gpu:
            speedup = cpu["avg_ms"] / gpu["avg_ms"] if gpu["avg_ms"] else 0
            print(f"  GPU vs CPU ({name})         : x{speedup:.2f}")

    print("-" * 60 + "\n")


def save_json(results: list[dict],
              path: str = "data/benchmark_results.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    onnx_providers = []
    onnx_version = None
    try:
        import onnxruntime as ort
        onnx_providers = ort.get_available_providers()
        onnx_version = ort.__version__
    except Exception:
        pass

    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "onnxruntime": onnx_version,
            "onnx_providers": onnx_providers,
        },
        "config": {
            "imgsz": IMGSZ,
            "input_shape": INPUT_SHAPE,
            "warmup": WARMUP,
            "runs": RUNS,
        },
        "results": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Résultats sauvegardés dans {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\n🚀 Driver Monitoring — Benchmark")
    print("=" * 60)
    print(f"PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU     : non disponible")

    try:
        import onnxruntime as ort
        print(f"ONNX    : {ort.__version__}")
        print(f"Providers : {ort.get_available_providers()}")
    except Exception:
        pass

    print(f"Runs    : {RUNS}  |  Warmup : {WARMUP}  |  imgsz : {IMGSZ}")
    print("=" * 60)

    results: list[dict] = []

    # ---- PyTorch CPU ----
    if Path(PT_MODEL).exists():
        print(f"\n⏳ Benchmark {PT_MODEL} [CPU]...")
        results.append(benchmark(PT_MODEL, device="cpu"))

    # ---- PyTorch GPU ----
    if Path(PT_MODEL).exists() and torch.cuda.is_available():
        print(f"⏳ Benchmark {PT_MODEL} [GPU]...")
        results.append(benchmark(PT_MODEL, device="0"))

    # ---- ONNX CPU ----
    if Path(ONNX_MODEL).exists():
        print(f"⏳ Benchmark {ONNX_MODEL} [CPU]...")
        results.append(benchmark(ONNX_MODEL, device="cpu"))

    # ---- ONNX GPU ----
    if Path(ONNX_MODEL).exists() and torch.cuda.is_available():
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                print(f"⏳ Benchmark {ONNX_MODEL} [GPU]...")
                results.append(benchmark(ONNX_MODEL, device="0"))
            else:
                print("ℹ️  ONNX GPU ignoré (CUDAExecutionProvider absent)")
        except Exception as e:
            print(f"ℹ️  ONNX GPU ignoré ({e})")

    if not results:
        print("\n❌ Aucun modèle trouvé. Lance d'abord export_onnx.py")
        return

    print_table(results)
    print_speedups(results)
    save_json(results)


if __name__ == "__main__":
    main()