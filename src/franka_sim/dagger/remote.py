"""Run a policy in another process or on another machine (e.g. a GPU box) and use it from the sim.

The policy side wraps any benchmark-style policy factory::

    # on the GPU machine (needs only numpy + the server extra, no MuJoCo rendering)
    uv run --extra server python scripts/serve_policy.py --policy my_pkg.policies:make \\
        --policy-kwargs '{"ckpt": "runs/act_round2"}' --host 0.0.0.0 --port 9000

and the simulation side uses ``RemotePolicy`` wherever a policy factory is expected::

    factory = remote_factory("ws://gpu-box:9000")          # factory(env) -> RemotePolicy
    uv run python scripts/evaluate.py --policy remote:ws://gpu-box:9000
    uv run --extra teleop python scripts/dagger.py teleop --policy remote:ws://gpu-box:9000

Protocol (msgpack over WebSocket, see ``franka_sim.server.protocol``): the client first sends
``{"op": "init", "spec": {...}}`` with the env's facts (task, action mode and dimension, batch size,
cameras, image size, control period), and the server builds the policy with
``factory(EnvSpec(**spec), **kwargs)``. Then ``reset`` (``env_ids``, ``obs``), ``interrupt`` (same)
and ``act`` (``obs``) forward to the policy. Each connection gets its own policy instance.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from franka_sim.server.protocol import pack, unpack


@dataclass
class EnvSpec:
    """The facts about the env a remote policy factory gets in place of the ``FrankaEnv``."""

    task: str
    action_mode: str
    action_dim: int
    num_envs: int
    cameras: list = field(default_factory=list)
    image_size: list = field(default_factory=list)
    control_dt: float = 0.05
    instruction_example: str = ""

    @classmethod
    def from_env(cls, env) -> EnvSpec:
        return cls(task=env.task.name, action_mode=env.action_mode, action_dim=env.action_dim,
                   num_envs=env.num_envs, cameras=list(env.cameras), image_size=list(env.image_size),
                   control_dt=env.control_dt, instruction_example=env.instructions()[0])


def load_factory(spec: str, kwargs: dict | None = None) -> Callable:
    """``"module.path:factory"``, ``"scripted"`` or ``"remote:ws://host:port"`` -> ``factory(env)``."""
    kwargs = dict(kwargs or {})
    if spec == "scripted":
        from franka_sim.policies import make_policy

        return partial(make_policy, **kwargs)
    if spec.startswith("remote:"):
        return remote_factory(spec[len("remote:"):], **kwargs)
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError(f"policy must be 'scripted', 'remote:ws://...' or 'module.path:factory', got {spec!r}")
    return partial(getattr(importlib.import_module(module), name), **kwargs)


class RemotePolicy:
    """Benchmark-style policy (``reset(env_ids, obs)`` / ``act(obs)``) served by ``serve_policy``."""

    def __init__(self, url: str, env, timeout: float = 120.0):
        from contextlib import ExitStack

        from websockets.sync.client import connect

        self.url = url
        self.timeout = timeout
        self._stack = ExitStack()
        self._ws = self._stack.enter_context(connect(url, max_size=256 << 20, open_timeout=timeout))
        spec = EnvSpec.from_env(env).__dict__
        info = self._call("init", spec=spec)
        self.action_mode = info.get("action_mode") or env.action_mode
        if self.action_mode != env.action_mode:
            raise ValueError(f"the remote policy emits {self.action_mode} actions but the env uses {env.action_mode}")
        self.action_dim = env.action_dim
        self.num_envs = env.num_envs

    def _call(self, op: str, **kwargs):
        self._ws.send(pack({"op": op, **kwargs}))
        reply = unpack(self._ws.recv(timeout=self.timeout))
        if not reply["ok"]:
            raise RuntimeError(f"remote policy: {reply['error']}")
        return reply["result"]

    def reset(self, env_ids, obs) -> None:
        self._call("reset", env_ids=np.asarray(env_ids), obs=obs)

    def interrupt(self, env_ids, obs) -> None:
        """A human took over or handed back mid-episode: drop queued actions (see ``dagger``)."""
        self._call("interrupt", env_ids=np.asarray(env_ids), obs=obs)

    def act(self, obs) -> np.ndarray:
        out = np.asarray(self._call("act", obs=obs), dtype=np.float64)
        return out.reshape(self.num_envs, self.action_dim)

    def close(self) -> None:
        self._stack.close()


def remote_factory(url: str, **kwargs) -> Callable:
    return lambda env: RemotePolicy(url, env, **kwargs)


def create_policy_app(factory: Callable):
    """FastAPI app serving ``factory``-built policies on ``/`` (one instance per connection)."""
    from franka_sim.dagger.policy_server import create_app

    return create_app(factory)


def serve_policy(factory: Callable | str, host: str = "127.0.0.1", port: int = 9000,
                 factory_kwargs: dict | None = None) -> None:
    """Serve a policy factory (callable or ``"module:factory"`` string) until interrupted."""
    import uvicorn

    if isinstance(factory, str):
        factory = load_factory(factory, factory_kwargs)
    elif factory_kwargs:
        factory = partial(factory, **factory_kwargs)
    uvicorn.run(create_policy_app(factory), host=host, port=port, log_level="warning", ws_max_size=256 << 20)


__all__ = ["EnvSpec", "RemotePolicy", "create_policy_app", "load_factory", "remote_factory", "serve_policy"]
