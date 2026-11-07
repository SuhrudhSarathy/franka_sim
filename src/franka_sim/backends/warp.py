"""MuJoCo Warp backend: all envs batched as ``nworld`` worlds on one device (CUDA GPU, or CPU)."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np

try:
    import mujoco_warp as mjw
    import warp as wp
except ImportError as e:  # pragma: no cover - exercised only without the extra installed
    raise ImportError("the warp backend needs the optional dependency: `uv sync --extra warp`") from e

from franka_sim.backends.base import Backend, as_env_ids

wp.config.quiet = True


class WarpBackend(Backend):
    """Batched MuJoCo Warp simulation.

    Args:
        model: host model.
        num_envs: number of worlds.
        device: Warp device (``"cuda:0"``, ``"cpu"``); defaults to the preferred device.
        nconmax: contacts allocated per world.
        njmax: constraint rows allocated per world.
        use_graph: capture the substep loop in a CUDA graph (CUDA devices only).
    """

    name = "warp"

    def __init__(self, model: mujoco.MjModel, num_envs: int = 1, device: str | None = None,
                 nconmax: int = 64, njmax: int = 256, use_graph: bool = True):
        super().__init__(model, num_envs)
        wp.init()
        self.device = wp.get_device(device) if device is not None else wp.get_preferred_device()
        self.use_graph = use_graph and self.device.is_cuda
        self._graphs: dict[int, object] = {}
        self._reset_mask = np.zeros(num_envs, dtype=bool)
        host = mujoco.MjData(model)
        mujoco.mj_forward(model, host)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model)
            self.d = mjw.put_data(model, host, nworld=num_envs, nconmax=nconmax, njmax=njmax)
        self._shapes = {name: np.array(getattr(host, name)).shape for name in dir(host)
                        if isinstance(getattr(host, name, None), np.ndarray)}

    # ------------------------------------------------------------------------------------------
    def _field(self, name: str):
        try:
            return getattr(self.d, name)
        except AttributeError:
            raise AttributeError(f"MuJoCo Warp Data has no field {name!r}") from None

    def read(self, field: str) -> np.ndarray:
        # On CPU devices ``numpy()`` is a zero-copy view of the simulation buffer: copy it.
        arr = np.array(self._field(field).numpy(), copy=True)
        shape = self._shapes.get(field)
        if shape is not None:
            arr = arr.reshape((self.num_envs,) + shape)
        return arr

    def write(self, field: str, values: np.ndarray, env_ids=None) -> None:
        arr = self._field(field)
        ids = as_env_ids(env_ids, self.num_envs)
        host = arr.numpy()
        values = np.asarray(values, dtype=host.dtype)
        host[ids] = values.reshape((len(ids),) + host.shape[1:])
        arr.assign(host)

    def reset(self, env_ids=None) -> None:
        mask = self._reset_mask
        mask[:] = False
        mask[as_env_ids(env_ids, self.num_envs)] = True
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.m, self.d, reset=wp.array(mask, dtype=bool))

    def forward(self) -> None:
        with wp.ScopedDevice(self.device):
            mjw.forward(self.m, self.d)

    def _substeps(self, n: int) -> None:
        for _ in range(n):
            mjw.step(self.m, self.d)

    def step(self, ctrl: np.ndarray, nsubsteps: int = 1) -> None:
        ctrl = np.broadcast_to(np.asarray(ctrl, dtype=np.float32), (self.num_envs, self.model.nu))
        self.d.ctrl.assign(np.ascontiguousarray(ctrl))
        if self.use_graph and nsubsteps not in self._graphs:
            self.warmup(nsubsteps)
        with wp.ScopedDevice(self.device):
            if self.use_graph:
                wp.capture_launch(self._graphs[nsubsteps])
            else:
                self._substeps(nsubsteps)
        self._check_overflow()

    def warmup(self, nsubsteps: int) -> None:
        """Compile kernels (and capture the CUDA graph) without disturbing the current state."""
        state = {f: self.read(f) for f in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "time")}
        with wp.ScopedDevice(self.device):
            if self.use_graph and nsubsteps not in self._graphs:
                self._substeps(nsubsteps)
                with wp.ScopedCapture() as capture:
                    self._substeps(nsubsteps)
                self._graphs[nsubsteps] = capture.graph
            elif not self.use_graph:
                self._substeps(1)
        self.reset()
        for f, v in state.items():
            self.write(f, v)
        self.forward()

    def _check_overflow(self) -> None:
        nacon = int(self.d.nacon.numpy()[0])
        if nacon >= self.d.naconmax:
            warnings.warn(f"MuJoCo Warp contact buffer full ({nacon}/{self.d.naconmax}); "
                          "increase backend_kwargs['nconmax']", RuntimeWarning, stacklevel=3)

    def sync_mjdata(self, env_id: int, data: mujoco.MjData) -> mujoco.MjData:
        mjw.get_data_into(data, self.model, self.d, world_id=env_id)
        mujoco.mj_forward(self.model, data)
        return data
