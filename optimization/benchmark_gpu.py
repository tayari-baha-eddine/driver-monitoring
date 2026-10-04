"""Benchmark YOLOv8n CPU vs GPU with diagnostics."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO


def diagnose() -> None:
    print("=== DIAGNOSTIC ===")
    print(f"Python         : {sys.version.split()[0]}")
    print(f"torch version  : {torch.__version__}")
    print(f"torch file     : {torch.__file__}")
    print(f"CUDA compiled  : {torch.version.cuda}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Device count   : {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    if torch.cuda.is_available():
        print(f"GPU 0          : {torch.cuda.get_device_name(0)}")
    print("=" * 40 + "\n")


def benchmark(device: str, n: int = 50, warmup: int = 5) -> dict:
    model = YOLO("models/pytorch/yolov8n.pt")
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(frame, device=device, verbose=False)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(frame, device=device, verbose=False)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    p95 = sorted(times)[int(0.95 * len(times))]
    return {
        "device": device,
        "avg_ms": round(avg * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "fps": round(1 / avg, 1),
    }


def main() -> None:
    diagnose()
    print("=== Benchmark YOLOv8n ===")

    cpu = benchmark("cpu")
    print(f"CPU : {cpu['avg_ms']:>7.2f} ms (p95 {cpu['p95_ms']:.1f})  ->  {cpu['fps']:>5.1f} FPS")

    if torch.cuda.is_available():
        try:
            gpu = benchmark("0")
            print(f"GPU : {gpu['avg_ms']:>7.2f} ms (p95 {gpu['p95_ms']:.1f})  ->  {gpu['fps']:>5.1f} FPS")
            print(f"\nSpeedup GPU : x{gpu['fps'] / cpu['fps']:.1f}")
            print(f"VRAM used   : {round(torch.cuda.memory_allocated() / 1024 / 1024, 1)} MB")
        except Exception as e:
            print(f"[GPU ERROR] {type(e).__name__}: {e}")
    else:
        print("WARNING: CUDA not available according to torch.cuda.is_available()")


if __name__ == "__main__":
    main()