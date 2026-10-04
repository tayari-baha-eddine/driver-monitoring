"""Export YOLOv8 PyTorch -> ONNX for edge deployment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_onnx(
    model_path: str,
    output_dir: str = "models/onnx",
    imgsz: int = 640,
    opset: int = 12,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """
    Export a YOLOv8 .pt model to ONNX format.

    Args:
        model_path: Path to the .pt file.
        output_dir: Where to place the .onnx (moved after export).
        imgsz: Input image size (square).
        opset: ONNX opset version.
        half: FP16 export (smaller, faster on GPU, less compatible).
        dynamic: Dynamic input shapes (slightly slower inference).
        simplify: Run onnx-simplifier to fuse redundant ops.

    Returns:
        Path to the final .onnx file.
    """
    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")

    print(f"[1/3] Loading {src}")
    model = YOLO(str(src))

    print(f"[2/3] Exporting to ONNX (imgsz={imgsz}, opset={opset}, "
          f"half={half}, dynamic={dynamic}, simplify={simplify})")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
    )

    exported_path = Path(exported)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / exported_path.name

    if exported_path.resolve() != final_path.resolve():
        shutil.move(str(exported_path), str(final_path))

    size_mb = final_path.stat().st_size / 1024 / 1024
    print(f"[3/3] Done: {final_path} ({size_mb:.2f} MB)")
    return final_path


def main() -> None:
    p = argparse.ArgumentParser(description="YOLOv8 -> ONNX export")
    p.add_argument("--model", default="models/pytorch/yolov8n.pt")
    p.add_argument("--output-dir", default="models/onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--half", action="store_true",
                   help="FP16 (GPU-friendly)")
    p.add_argument("--dynamic", action="store_true",
                   help="Dynamic input shapes")
    p.add_argument("--no-simplify", action="store_true")
    args = p.parse_args()

    export_onnx(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()