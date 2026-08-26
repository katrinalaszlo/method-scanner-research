"""Phase 23 shim: rendering is disabled; MuJoCoRenderer catches the failure and sets viewer=None."""
class MjRenderContextOffscreen:
    def __init__(self, *a, **k):
        raise RuntimeError("mujoco_py shim: no offscreen rendering")
