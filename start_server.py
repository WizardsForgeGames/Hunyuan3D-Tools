"""
Start the Hunyuan3D API server.

Usage:
    python start_server.py                  # Default settings
    python start_server.py --low-vram       # Offload between stages (saves VRAM)
    python start_server.py --port 9090      # Custom port

Once running, generate meshes with:
    python client.py --image photo.png
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")
# Use venv Python if it exists, otherwise fall back to current interpreter
_venv_python = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
VENV_PYTHON = _venv_python if os.path.isfile(_venv_python) else sys.executable


def main():
    parser = argparse.ArgumentParser(description="Start Hunyuan3D API server")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1", help="Model path")
    args = parser.parse_args()

    cmd = [
        VENV_PYTHON,
        os.path.join(REPO_DIR, "api_server.py"),
        "--host", args.host,
        "--port", str(args.port),
        "--model_path", args.model_path,
        "--subfolder", "hunyuan3d-dit-v2-1",
    ]

    if args.low_vram:
        cmd.append("--low_vram_mode")

    print(f"Starting Hunyuan3D server on {args.host}:{args.port}")
    print(f"Model: {args.model_path}")
    print(f"Low VRAM: {args.low_vram}")
    print()
    print("The first start downloads model weights (~6GB). Subsequent starts use cached weights.")
    print("Press Ctrl+C to stop.\n")

    os.chdir(REPO_DIR)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
