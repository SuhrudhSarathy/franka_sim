"""The policy-serving web app (see ``franka_sim.dagger.remote``). Needs the ``server`` extra."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from franka_sim.server.protocol import pack, unpack


def create_app(factory: Callable) -> FastAPI:
    from franka_sim.dagger.remote import EnvSpec

    app = FastAPI(docs_url=None, redoc_url=None)

    @app.websocket("/")
    async def ws(socket: WebSocket):
        await socket.accept()
        policy = None
        try:
            while True:
                msg = unpack(await socket.receive_bytes())
                try:
                    op = msg["op"]
                    if op == "init":
                        policy = await asyncio.to_thread(factory, EnvSpec(**msg["spec"]))
                        result = {"action_mode": getattr(policy, "action_mode", None)}
                    elif policy is None:
                        raise RuntimeError("send 'init' first")
                    elif op == "act":
                        result = np.asarray(await asyncio.to_thread(policy.act, msg["obs"]))
                    elif op == "reset":
                        result = await asyncio.to_thread(policy.reset, msg["env_ids"], msg["obs"])
                    elif op == "interrupt":
                        fn = getattr(policy, "interrupt", policy.reset)
                        result = await asyncio.to_thread(fn, msg["env_ids"], msg["obs"])
                    else:
                        raise ValueError(f"unknown op {op!r}")
                    reply = {"ok": True, "result": result}
                except Exception as e:  # report to the simulation, keep the connection
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                await socket.send_bytes(pack(reply))
        except WebSocketDisconnect:
            pass

    return app
