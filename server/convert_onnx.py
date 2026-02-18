"""
Standalone ONNX model converter for SpriteGen.

Usage:
    python convert_onnx.py <model_name> <output_dir> [dtype]

Example:
    python convert_onnx.py stabilityai/stable-diffusion-xl-base-1.0 onnx_models/stabilityai--stable-diffusion-xl-base-1.0 fp16

Uses optimum.exporters (NOT optimum.onnxruntime) to avoid the onnxruntime
namespace conflict with onnxruntime-directml.
"""
import sys
import os

VALID_DTYPES = ("fp16", "bf16", "fp32")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_name> <output_dir> [dtype]", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    output_dir = sys.argv[2]
    dtype = sys.argv[3] if len(sys.argv) > 3 else "fp16"

    if dtype not in VALID_DTYPES:
        print(f"Invalid dtype '{dtype}'. Must be one of: {', '.join(VALID_DTYPES)}", file=sys.stderr)
        sys.exit(1)

    # Skip if already converted
    index_path = os.path.join(output_dir, "model_index.json")
    if os.path.isfile(index_path):
        print(f"ONNX model already exists at '{output_dir}', skipping conversion.")
        sys.exit(0)

    print(f"Converting '{model_name}' to ONNX format ({dtype.upper()})...")
    print(f"Output directory: {output_dir}")
    print("This may take several minutes on first run.")

    # Patch: optimum's allowlist of onnxruntime distributions doesn't include
    # onnxruntime-directml. The module IS importable, so we tell optimum it's available.
    import optimum.utils.import_utils as _import_utils
    _import_utils._onnxruntime_available = True

    from optimum.exporters.onnx import main_export

    main_export(
        model_name_or_path=model_name,
        output=output_dir,
        dtype=dtype,
    )

    if os.path.isfile(index_path):
        print(f"Conversion complete! ONNX model saved to '{output_dir}'.")
    else:
        print("Warning: Conversion finished but model_index.json not found.")
        print("The model may still be usable. Check the output directory.")


if __name__ == "__main__":
    main()
