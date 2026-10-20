"""
Remesh + UV-unwrap meshes in a worker process, ahead of the texture pipeline.

Remeshing (pymeshlab) and UV unwrapping (xatlas) are CPU work that depends only
on the input mesh, so it can run while the texture models load (texture.py) or
while the previous mesh is being textured on the GPU (texture_batch.py). The
worker runs the same upstream functions as Hunyuan3DPaintPipeline.__call__ in
the same order, so the prepared mesh is identical; pass it to the pipeline as
`mesh=`.

The worker imports only pymeshlab / trimesh / xatlas (no torch), so it starts
in about a second.
"""

import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")


def _init_worker():
    sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))
    sys.path.insert(0, SCRIPT_DIR)
    from win_qos import request_high_qos
    request_high_qos()


def prepare_mesh(mesh_path, use_remesh=True):
    """Remesh (optional) and UV-unwrap `mesh_path`, as the pipeline would. Returns a trimesh."""
    import trimesh
    from utils.simplify_mesh_utils import remesh_mesh
    from utils.uvwrap_utils import mesh_uv_wrap

    if not use_remesh:
        return mesh_uv_wrap(trimesh.load(mesh_path))
    # The pipeline writes white_mesh_remesh.obj next to the input mesh; a private
    # directory keeps prefetched meshes from overwriting each other.
    work_dir = tempfile.mkdtemp(prefix="hy3d_remesh_")
    try:
        remesh_path = os.path.join(work_dir, "white_mesh_remesh.obj")
        remesh_mesh(mesh_path, remesh_path)
        return mesh_uv_wrap(trimesh.load(remesh_path))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


class MeshPrefetcher:
    """One worker process that prepares meshes in the background, in submission order."""

    def __init__(self):
        self._pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)

    def submit(self, mesh_path, use_remesh=True):
        """Start preparing `mesh_path`; returns a Future whose result() is the trimesh."""
        return self._pool.submit(prepare_mesh, mesh_path, use_remesh)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
