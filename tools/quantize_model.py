"""Create a calibrated INT8 ONNX model for browser CPU inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, image_size: int = 640):
        self.input_name = input_name
        self.samples = iter(self._prepare(path, image_size) for path in image_paths)

    @staticmethod
    def _prepare(path: Path, image_size: int) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read calibration image: {path}")

        height, width = image.shape[:2]
        scale = min(image_size / width, image_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
        pad_x = (image_size - resized_width) // 2
        pad_y = (image_size - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {self.input_name: next(self.samples)}
        except StopIteration:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("best.onnx"))
    parser.add_argument("--output", type=Path, default=Path("best.int8.onnx"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument(
        "--max-layer",
        type=int,
        default=15,
        help="Last YOLO layer to quantize; later detection-sensitive layers remain float.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in extensions)
    if not image_paths:
        raise SystemExit(f"No calibration images found in {args.images}")

    model = onnx.load(args.input, load_external_data=False)
    input_name = model.graph.input[0].name
    reader = ImageCalibrationReader(image_paths, input_name)
    def should_remain_float(node: onnx.NodeProto) -> bool:
        match = re.match(r"/model\.(\d+)(?:/|$)", node.name)
        return bool(match and int(match.group(1)) > args.max_layer)

    float_nodes = [node.name for node in model.graph.node if should_remain_float(node)]

    print(f"Calibrating with {len(image_paths)} images from {args.images}")
    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        # Opset 12 does not support the axis attribute used by per-channel QDQ.
        per_channel=False,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=float_nodes,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": False},
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
