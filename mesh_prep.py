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
import multiprocessing

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


class _Pending:
    """Handle for one submitted mesh; result() waits for it (Future-like)."""

    def __init__(self, async_result):
        self._r = async_result

    def result(self, timeout=None):
        return self._r.get(timeout)


class MeshPrefetcher:
    """One worker process that prepares meshes in the background, in submission order.

    Uses a multiprocessing.Pool, not a ProcessPoolExecutor: Pool workers are daemon
    processes that are terminated when the caller exits, whereas concurrent.futures
    waits at interpreter exit for every queued mesh. If the caller fails (e.g. a
    torch.load error while the pipeline loads) that wait delayed its exit, and
    batch_demo.py's retry, by ~9 s per queued mesh.
    """

    def __init__(self):
        self._pool = multiprocessing.get_context("spawn").Pool(1, initializer=_init_worker)

    def submit(self, mesh_path, use_remesh=True):
        """Start preparing `mesh_path`; returns a handle whose result() is the trimesh."""
        return _Pending(self._pool.apply_async(prepare_mesh, (mesh_path, use_remesh)))

    def close(self):
        """Stop now: drop queued meshes and kill a mesh in progress."""
        self._pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
