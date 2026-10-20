"""
Hunyuan3D-2.1 Full Pipeline Batch Demo
Shape generation + optimized texture painting on RTX 4090.

Runs shape gen and texture gen as two separate processes (one each, so each model
loads once per batch) to avoid import conflicts between hy3dshape and hy3dpaint.

Usage:
    python batch_demo.py                         # 10 default samples
    python batch_demo.py --compile               # + torch.compile UNet
    python batch_demo.py --images img1.png img2.png  # custom images
    python batch_demo.py --skip-shape            # texture only (reuse meshes)
"""

import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")

# Run at High QoS so Windows keeps the CPU stages on the P-cores (see win_qos.py)
from win_qos import request_high_qos
request_high_qos()

# Default showcase images - 10 varied subjects
DEFAULT_IMAGES = [
    "assets/example_images/ainimal_9.png",
    "assets/example_images/ainimal_14.png",
    "assets/example_images/004.png",
    "assets/example_images/073.png",
    "assets/example_images/202.png",
    "assets/example_images/419.png",
    "assets/example_images/895.png",
    "assets/example_images/jimeng2.png",
    "assets/example_images/jimeng42.png",
    "assets/example_images/example_000.png",
]


def run_shape_batch(jobs, steps=50, seed=None):
    """Generate all shapes in ONE subprocess, so the shape model loads once per batch.

    jobs: [(image_path, output_mesh), ...]. Returns ({index: time or -1}, load_time).
    Output is streamed as it arrives.
    """
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "generate_shape.py"),
        "--pairs", *[f"{img}|{out}" for img, out in jobs],
        "--steps", str(steps),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    times = {}
    load_time = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        m = re.match(r"SHAPE_TIME:(\d+)=([\d.]+|FAIL)$", line)
        if m:
            times[int(m.group(1))] = -1 if m.group(2) == "FAIL" else float(m.group(2))
            continue
        m = re.match(r"SHAPE_LOAD=([\d.]+)$", line)
        if m:
            load_time = float(m.group(1))
            continue
        tail = (tail + [line])[-5:]
        print(f"    {line}", flush=True)
    proc.wait()
    if proc.returncode != 0 and len(times) < len(jobs):
        print("    ERROR: shape gen failed")
        for line in tail:
            print(f"    ! {line}")
    for i in range(len(jobs)):
        times.setdefault(i, -1)
    return times, load_time


def run_texture(mesh_path, image_path, output_path, quantize_mode="tensorcore", compile_flag=False, texture_steps=15, max_retries=2):
    """Run texture generation as a subprocess. Returns (time, success).
    Retries on intermittent PyTorch PytorchStreamReader corruption (PyTorch 2.8 bug)."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "texture.py"),
        "--mesh", mesh_path,
        "--image", image_path,
        "--output", output_path,
        "--quantize-mode", quantize_mode,
        "--steps", str(texture_steps),
    ]
    if compile_flag:
        cmd.append("--compile")

    for attempt in range(max_retries + 1):
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall_time = time.time() - t0

        # Check for intermittent torch.load corruption — retry if so
        if result.returncode != 0 and "PytorchStreamReader" in (result.stderr or ""):
            if attempt < max_retries:
                print(f"    (torch.load glitch, retry {attempt+1}/{max_retries})")
                continue

        break

    # Print subprocess output (just the timing lines)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith("[") or "generated in" in line.lower() or "Done!" in line:
                print(f"    {line.strip()}")

    if result.returncode != 0:
        print(f"    ERROR: texture gen failed")
        if result.stderr:
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"    ! {line}")
        return wall_time, False

    # Parse timing from "Textures generated in XX.Xs"
    match = re.search(r"Textures generated in ([\d.]+)s", result.stdout)
    gen_time = float(match.group(1)) if match else wall_time

    return gen_time, True


def main():
    parser = argparse.ArgumentParser(description="Batch shape+texture demo")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Custom input images (default: 10 built-in examples)")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of default samples to run (max 10)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the texture UNet")
    parser.add_argument("--quantize-mode", default="tensorcore",
                        choices=["sparse", "tensorcore", "none"])
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "output", "samples"))
    parser.add_argument("--skip-shape", action="store_true",
                        help="Skip shape gen (use pre-existing meshes in output dir)")
    parser.add_argument("--shape-steps", type=int, default=50,
                        help="Shape diffusion steps (default: 50)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed each shape's initial noise (default: unseeded)")
    parser.add_argument("--texture-steps", type=int, default=15,
                        help="Texture denoising steps (default: 15, try 10 for speed)")
    args = parser.parse_args()

    # Resolve image paths
    if args.images:
        image_paths = [os.path.abspath(p) for p in args.images]
    else:
        image_paths = [os.path.join(REPO_DIR, p) for p in DEFAULT_IMAGES[:args.num]]

    # Validate
    for p in image_paths:
        if not os.path.isfile(p):
            print(f"Error: image not found: {p}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    n = len(image_paths)

    print("=" * 60)
    print(f"  Hunyuan3D Batch Demo - {n} models")
    print(f"  Output: {args.output_dir}")
    print(f"  Quantize: {args.quantize_mode}  Compile: {args.compile}")
    print("=" * 60)
    print()

    shape_times = {}
    texture_times = {}
    names = []

    # ═══════════════════════════════════════════
    #  Phase 1: Shape Generation (one process, model loaded once)
    # ═══════════════════════════════════════════
    mesh_paths = {}
    t_batch = time.time()
    shape_wall = texture_wall = None
    if not args.skip_shape:
        print("--- PHASE 1: Shape Generation (batch) ---")
        print()
        jobs = []
        for img_path in image_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            names.append(name)
            out_mesh = os.path.join(args.output_dir, f"{name}_mesh.glb")
            mesh_paths[name] = out_mesh
            jobs.append((img_path, out_mesh))

        t_phase = time.time()
        times, shape_load = run_shape_batch(jobs, steps=args.shape_steps, seed=args.seed)
        shape_wall = time.time() - t_phase
        for idx, name in enumerate(names):
            shape_times[name] = times[idx]
            print(f"  {name}: " + (f"{times[idx]:.1f}s" if times[idx] > 0 else "FAILED"))
        if shape_load is not None:
            print(f"  (shape model load: {shape_load:.1f}s, once)")
        print()
    else:
        for img_path in image_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            names.append(name)
            out_mesh = os.path.join(args.output_dir, f"{name}_mesh.glb")
            if not os.path.isfile(out_mesh):
                print(f"Error: --skip-shape but mesh not found: {out_mesh}")
                sys.exit(1)
            mesh_paths[name] = out_mesh
        print("--- Skipping shape generation (using existing meshes) ---")
        print()

    # ═══════════════════════════════════════════
    #  Phase 2: Texture Generation (single process, load once)
    # ═══════════════════════════════════════════
    print("--- PHASE 2: Texture Generation (batch) ---")
    print()

    # Build pairs for texture_batch.py
    batch_pairs = []
    for name in names:
        mesh_path = mesh_paths[name]
        if not os.path.isfile(mesh_path):
            print(f"  {name} - skipped (no mesh)")
            texture_times[name] = -1
            continue
        img_path = image_paths[names.index(name)]
        batch_pairs.append(f"{mesh_path}|{img_path}")

    t_phase = time.time()
    if batch_pairs:
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "texture_batch.py"),
            "--pairs", *batch_pairs,
            "--output-dir", args.output_dir,
            "--quantize-mode", args.quantize_mode,
            "--steps", str(args.texture_steps),
        ]
        if args.compile:
            cmd.append("--compile")

        max_retries = 3
        for attempt in range(max_retries):
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")

            # Retry on intermittent PyTorch 2.8 torch.load corruption
            if result.returncode != 0 and "PytorchStreamReader" in (result.stderr or ""):
                if attempt < max_retries - 1:
                    print(f"  (torch.load corruption - retry {attempt+1}/{max_retries-1})")
                    continue
            break

        # Print output
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if not line.startswith("TEXTURE_TIME:"):
                    print(f"  {line}")

            # Parse per-model times
            for line in result.stdout.strip().split("\n"):
                if line.startswith("TEXTURE_TIME:"):
                    parts = line[len("TEXTURE_TIME:"):].split("=")
                    name_key = parts[0]
                    val = parts[1]
                    if val == "FAIL":
                        texture_times[name_key] = -1
                    else:
                        texture_times[name_key] = float(val)

        if result.returncode != 0 and result.stderr:
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-10:]:
                print(f"  ! {line}")

    texture_wall = time.time() - t_phase

    # ═══════════════════════════════════════════
    #  Results
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()
    print(f"  {'Sample':<25s} {'Shape':>8s} {'Texture':>8s} {'Total':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    valid_shape = []
    valid_texture = []
    for name in names:
        t_s = shape_times.get(name, 0)
        t_t = texture_times.get(name, 0)
        s_str = f"{t_s:.1f}s" if t_s > 0 else "FAIL" if t_s < 0 else "skip"
        t_str = f"{t_t:.1f}s" if t_t > 0 else "FAIL"
        total = (t_s if t_s > 0 else 0) + (t_t if t_t > 0 else 0)
        total_str = f"{total:.1f}s" if t_t > 0 else "-"
        print(f"  {name:<25s} {s_str:>8s} {t_str:>8s} {total_str:>8s}")
        if t_s > 0:
            valid_shape.append(t_s)
        if t_t > 0:
            valid_texture.append(t_t)

    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    if valid_shape or valid_texture:
        avg_s = sum(valid_shape) / len(valid_shape) if valid_shape else 0
        avg_t = sum(valid_texture) / len(valid_texture) if valid_texture else 0
        print(f"  {'Average':<25s} {avg_s:>7.1f}s {avg_t:>7.1f}s {avg_s+avg_t:>7.1f}s")

    print()
    if shape_wall is not None:
        print(f"  Shape phase wall time:   {shape_wall:.1f}s")
    print(f"  Texture phase wall time: {texture_wall:.1f}s")
    print(f"  Total wall time:         {time.time() - t_batch:.1f}s")
    print(f"  Output: {args.output_dir}")
    succeeded = sum(1 for t in valid_texture if t > 0)
    print(f"  Completed: {succeeded}/{n}")


if __name__ == "__main__":
    main()
