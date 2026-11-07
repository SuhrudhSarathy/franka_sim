"""Common interface for batched physics backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np


class Backend(ABC):
    """A batch of ``num_envs`` independent worlds sharing one ``MjModel``.

    Every array read or written through this interface is a numpy array with a leading
    ``num_envs`` (or ``len(env_ids)``) axis followed by the shape the field has in ``mujoco.MjData``
    (e.g. ``site_xmat`` is ``(N, nsite, 9)``).
    """

    name: str

    def __init__(self, model: mujoco.MjModel, num_envs: int):
        self.model = model
        self.num_envs = num_envs
        self._scratch = mujoco.MjData(model)

    @abstractmethod
    def read(self, field: str) -> np.ndarray:
        """Return a copy of an ``MjData`` field for all envs, shape ``(N, *field_shape)``."""

    @abstractmethod
    def write(self, field: str, values: np.ndarray, env_ids: np.ndarray | None = None) -> None:
        """Overwrite an ``MjData`` field for ``env_ids`` (all envs if ``None``)."""

    @abstractmethod
    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset ``env_ids`` to the model defaults (``qpos0``, zero velocity, time 0)."""

    @abstractmethod
    def forward(self) -> None:
        """Recompute derived quantities (kinematics, contacts, ...) without advancing time."""

    @abstractmethod
    def step(self, ctrl: np.ndarray, nsubsteps: int = 1) -> None:
        """Set ``ctrl`` (``(N, nu)``) and advance every env by ``nsubsteps`` physics steps."""

    @abstractmethod
    def sync_mjdata(self, env_id: int, data: mujoco.MjData) -> mujoco.MjData:
        """Copy the state of ``env_id`` into a host ``MjData`` (for rendering, IK, debugging)."""

    def mjdata(self, env_id: int = 0) -> mujoco.MjData:
        """A host ``MjData`` mirroring ``env_id``. The returned object is reused between calls."""
        return self.sync_mjdata(env_id, self._scratch)

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def as_env_ids(env_ids, num_envs: int) -> np.ndarray:
    if env_ids is None:
        return np.arange(num_envs)
    return np.atleast_1d(np.asarray(env_ids, dtype=np.int64))
