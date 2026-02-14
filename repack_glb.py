"""
Re-export textured OBJ files as proper glTF GLB with embedded PBR textures.

The default trimesh OBJ->GLB conversion only embeds the albedo (map_Kd).
This script properly embeds metallic and roughness maps as a combined
metallicRoughnessTexture per the glTF 2.0 PBR spec.

Usage:
    python repack_glb.py                                    # all in output/samples/
    python repack_glb.py --input output/samples/073_textured.obj
    python repack_glb.py --input-dir output/samples/ --output-dir output/repacked/
"""

import argparse
import glob
import io
import os
import struct
import sys

import numpy as np
import trimesh
from PIL import Image


def create_pbr_glb(obj_path, output_glb_path):
    """Load OBJ + separate PBR textures, export as a proper glTF GLB."""
    base = os.path.splitext(obj_path)[0]

    albedo_path = base + ".jpg"
    metallic_path = base + "_metallic.jpg"
    roughness_path = base + "_roughness.jpg"

    # Load mesh with trimesh (gets vertices, faces, UVs, albedo)
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        # Flatten scene to single mesh
        geom_list = list(mesh.geometry.values())
        if len(geom_list) == 1:
            mesh = geom_list[0]
        else:
            mesh = trimesh.util.concatenate(geom_list)

    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  Could not load as Trimesh: {type(mesh)}")
        return False

    # Load albedo texture
    if os.path.isfile(albedo_path):
        albedo = Image.open(albedo_path).convert("RGB")
    elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
        albedo = mesh.visual.material.baseColorTexture
        if albedo is not None:
            albedo = albedo.convert("RGB")
    else:
        albedo = None

    # Build metallicRoughness texture (glTF spec: B=metallic, G=roughness)
    mr_texture = None
    if os.path.isfile(metallic_path) and os.path.isfile(roughness_path):
        metallic = Image.open(metallic_path).convert("L")
        roughness = Image.open(roughness_path).convert("L")
        # Resize to match if needed
        size = metallic.size
        if roughness.size != size:
            roughness = roughness.resize(size, Image.LANCZOS)
        # glTF metallicRoughness: R=0, G=roughness, B=metallic, A=1
        mr_np = np.zeros((*size[::-1], 3), dtype=np.uint8)
        mr_np[:, :, 1] = np.array(roughness)  # G = roughness
        mr_np[:, :, 2] = np.array(metallic)   # B = metallic
        mr_texture = Image.fromarray(mr_np, "RGB")

    # Apply PBR material
    if albedo is not None:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo,
            metallicRoughnessTexture=mr_texture,
            metallicFactor=1.0 if mr_texture else 0.0,
            roughnessFactor=1.0 if mr_texture else 0.9,
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv if hasattr(mesh.visual, 'uv') else None,
            material=material,
        )

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_glb_path)), exist_ok=True)
    mesh.export(output_glb_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-export OBJ as proper PBR GLB")
    parser.add_argument("--input", default=None, help="Single OBJ file")
    parser.add_argument("--input-dir", default=None,
                        help="Directory of *_textured.obj files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input:
        objs = [args.input]
    elif args.input_dir:
        objs = sorted(glob.glob(os.path.join(args.input_dir, "*_textured.obj")))
    else:
        # Default: output/samples/
        objs = sorted(glob.glob(os.path.join(script_dir, "output", "samples", "*_textured.obj")))

    if not objs:
        print("No OBJ files found")
        sys.exit(1)

    print(f"Re-exporting {len(objs)} models as PBR GLB")
    print()

    for obj_path in objs:
        name = os.path.basename(obj_path).replace("_textured.obj", "")
        out_dir = args.output_dir or os.path.dirname(obj_path)
        out_path = os.path.join(out_dir, f"{name}_textured.glb")

        print(f"  {name}...", end=" ", flush=True)
        ok = create_pbr_glb(obj_path, out_path)
        if ok:
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"OK ({size_mb:.1f} MB)")
        else:
            print("FAILED")

    print("\nDone!")


if __name__ == "__main__":
    main()
