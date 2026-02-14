"""
Hunyuan3D API Client — Send generation requests to a running API server.

Start the server first:
    python start_server.py

Then generate:
    python client.py --image path/to/image.png
    python client.py --image image.png --output model.glb --texture
    python client.py --image image.png --steps 30 --seed 42

The server keeps models in VRAM, so subsequent requests are fast (~30-60s per mesh).
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

DEFAULT_SERVER = "http://localhost:8081"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send 3D generation requests to Hunyuan3D API server",
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default=None, help="Output .glb file path")
    parser.add_argument("--texture", "-t", action="store_true", help="Generate PBR textures")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Guidance scale (default: 5.0)")
    parser.add_argument("--octree-resolution", type=int, default=256, help="Octree resolution (default: 256)")
    parser.add_argument("--face-count", type=int, default=40000, help="Max face count for texturing (default: 40000)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"API server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async mode (poll for result)")
    return parser.parse_args()


def resolve_output_path(image_path, output_path, textured):
    if output_path:
        return output_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "output"), exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_textured" if textured else ""
    return os.path.join(script_dir, "output", f"{base}{suffix}.glb")


def check_server(server_url):
    try:
        req = urllib.request.Request(f"{server_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def generate_sync(server_url, image_b64, params):
    """Send a synchronous /generate request, returns GLB bytes."""
    payload = json.dumps({
        "image": image_b64,
        "texture": params.get("texture", False),
        "seed": params.get("seed", 1234),
        "num_inference_steps": params.get("steps", 50),
        "guidance_scale": params.get("guidance_scale", 5.0),
        "octree_resolution": params.get("octree_resolution", 256),
        "face_count": params.get("face_count", 40000),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        if resp.status == 200:
            return resp.read()
        else:
            body = resp.read().decode()
            raise RuntimeError(f"Server error {resp.status}: {body}")


def generate_async(server_url, image_b64, params):
    """Send an async /send request, poll /status until done, returns GLB bytes."""
    payload = json.dumps({
        "image": image_b64,
        "texture": params.get("texture", False),
        "seed": params.get("seed", 1234),
        "num_inference_steps": params.get("steps", 50),
        "guidance_scale": params.get("guidance_scale", 5.0),
        "octree_resolution": params.get("octree_resolution", 256),
        "face_count": params.get("face_count", 40000),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/send",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        uid = data["uid"]

    print(f"  Task submitted: {uid}")
    print(f"  Polling for result...", end="", flush=True)

    while True:
        time.sleep(3)
        req = urllib.request.Request(f"{server_url}/status/{uid}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        status = data.get("status")
        if status == "completed":
            print(" done!")
            return base64.b64decode(data["model_base64"])
        elif status == "error":
            raise RuntimeError(f"Generation failed: {data.get('message', 'unknown error')}")
        else:
            print(".", end="", flush=True)


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    output_path = resolve_output_path(args.image, args.output, args.texture)

    # Check server
    print(f"Connecting to {args.server}...")
    if not check_server(args.server):
        print(f"Error: Cannot reach server at {args.server}")
        print(f"Start the server first: python start_server.py")
        sys.exit(1)
    print("  Server is healthy.")

    # Encode image
    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "texture": args.texture,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "octree_resolution": args.octree_resolution,
        "face_count": args.face_count,
    }

    mode_str = "Shape + PBR Texture" if args.texture else "Shape only"
    print(f"Generating ({mode_str}, {args.steps} steps, seed {args.seed})...")
    t0 = time.time()

    if args.async_mode:
        glb_data = generate_async(args.server, image_b64, params)
    else:
        glb_data = generate_sync(args.server, image_b64, params)

    elapsed = time.time() - t0

    with open(output_path, "wb") as f:
        f.write(glb_data)

    size_mb = len(glb_data) / (1024 * 1024)
    print(f"\nDone! ({elapsed:.1f}s)")
    print(f"  Output: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
