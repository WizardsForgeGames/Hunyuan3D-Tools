"""
Render thumbnail screenshots of textured GLB models for documentation.

Usage:
    python render_screenshots.py                          # all *_textured.glb in output/samples/
    python render_screenshots.py --input model.glb        # single model
    python render_screenshots.py --grid                   # composite grid image
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import trimesh
from PIL import Image


def render_glb_to_image(glb_path, width=800, height=800):
    """Render a GLB file to a PIL Image using pyrender offscreen."""
    import pyrender

    # Load scene
    scene_or_mesh = trimesh.load(glb_path)

    # Create pyrender scene
    pr_scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.3, 0.3, 0.3])

    if isinstance(scene_or_mesh, trimesh.Scene):
        for name, geom in scene_or_mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(geom)
                pr_scene.add(mesh)
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(scene_or_mesh)
        pr_scene.add(mesh)

    # Compute bounding box from trimesh geometry (avoids pyrender numpy 2.0 bug)
    bounds = np.array([[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]])
    if isinstance(scene_or_mesh, trimesh.Scene):
        for geom in scene_or_mesh.geometry.values():
            if hasattr(geom, 'bounds') and geom.bounds is not None:
                bounds[0] = np.minimum(bounds[0], geom.bounds[0])
                bounds[1] = np.maximum(bounds[1], geom.bounds[1])
    elif hasattr(scene_or_mesh, 'bounds'):
        bounds = np.array(scene_or_mesh.bounds)

    center = (bounds[0] + bounds[1]) / 2.0
    size = np.linalg.norm(bounds[1] - bounds[0])
    distance = size * 1.5

    # Camera looking at center from front-right-above
    cam_pos = center + np.array([distance * 0.5, distance * 0.3, distance * 0.8])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    # Build camera pose (look-at)
    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos

    pr_scene.add(camera, pose=cam_pose)

    # Lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    pr_scene.add(light, pose=cam_pose)

    fill_pose = cam_pose.copy()
    fill_pose[:3, 3] = center + np.array([-distance * 0.5, distance * 0.2, -distance * 0.3])
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.9, 1.0], intensity=1.5)
    pr_scene.add(fill_light, pose=fill_pose)

    # Render
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(pr_scene)
        renderer.delete()
        return Image.fromarray(color)
    except Exception as e:
        print(f"  Render failed: {e}")
        return None


def make_grid(images, names, cols=5, thumb_size=400, padding=10, label_height=30):
    """Arrange images into a labeled grid."""
    rows = math.ceil(len(images) / cols)
    grid_w = cols * (thumb_size + padding) + padding
    grid_h = rows * (thumb_size + label_height + padding) + padding

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for idx, (img, name) in enumerate(zip(images, names)):
        row, col = divmod(idx, cols)
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_height + padding)

        thumb = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        grid.paste(thumb, (x, y))

        if draw and font:
            draw.text((x + 5, y + thumb_size + 2), name, fill=(0, 0, 0), font=font)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Render GLB screenshots")
    parser.add_argument("--input", default=None, help="Single GLB file")
    parser.add_argument("--input-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "samples"))
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "screenshots"))
    parser.add_argument("--grid", action="store_true", help="Also create composite grid")
    parser.add_argument("--size", type=int, default=800, help="Render resolution")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        glbs = [args.input]
    else:
        glbs = sorted(glob.glob(os.path.join(args.input_dir, "*_textured.glb")))

    if not glbs:
        print("No GLB files found")
        sys.exit(1)

    print(f"Rendering {len(glbs)} models to {args.output_dir}")
    images = []
    names = []

    for glb_path in glbs:
        name = os.path.basename(glb_path).replace("_textured.glb", "")
        out_path = os.path.join(args.output_dir, f"{name}.png")

        print(f"  {name}...", end=" ", flush=True)
        img = render_glb_to_image(glb_path, args.size, args.size)

        if img is not None:
            img.save(out_path)
            images.append(img)
            names.append(name)
            print(f"OK -> {out_path}")
        else:
            print("FAILED")

    if args.grid and len(images) >= 2:
        grid = make_grid(images, names)
        grid_path = os.path.join(args.output_dir, "grid.png")
        grid.save(grid_path)
        print(f"\nGrid saved to {grid_path}")

    print(f"\nDone! {len(images)}/{len(glbs)} rendered")


if __name__ == "__main__":
    main()
