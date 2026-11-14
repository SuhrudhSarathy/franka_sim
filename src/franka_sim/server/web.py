"""FastAPI app of ``RobotServer``: the model WebSocket and a JSON status endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from franka_sim.server import protocol

if TYPE_CHECKING:
    from franka_sim.server.app import RobotServer


def create_app(server: RobotServer) -> FastAPI:
    app = FastAPI(title="franka-sim robot", docs_url=None, redoc_url=None)
    robot = server.robot

    @app.get("/api/state")
    async def state():
        return server.state()

    def obs_args(msg: dict) -> dict:
        size = msg.get("image_size")
        return {"cameras": tuple(msg.get("cameras") or ()), "image_size": tuple(size) if size else None,
                "privileged": bool(msg.get("privileged", False))}

    async def handle(msg: dict):
        op = msg.get("op")
        if op == "info":
            out = await server.acall(robot.describe)
            out["running"] = server.running
            return out
        if op == "state":
            return server.state()
        if op == "observe":
            return await server.acall(robot.observation, **obs_args(msg))
        if op == "act":
            await server.acall(robot.command, msg["action"], msg.get("mode", "ee_pose"), msg.get("source", "model"))
            return None
        if op == "stop":
            return await server.acall(robot.stop)
        if op == "step":
            return await server.acall(server.lockstep, msg["action"], msg.get("mode", "ee_pose"),
                                      source=msg.get("source", "model"), **obs_args(msg))
        if op == "reset":
            return await server.acall(server.reset, msg.get("seed"), msg.get("options"), msg.get("task"),
                                      **obs_args(msg))
        if op in ("pause", "run"):
            return await server.acall(server.set_running, op == "run")
        raise ValueError(f"unknown op {op!r}")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        server.clients += 1
        try:
            while True:
                msg = protocol.unpack(await socket.receive_bytes())
                try:
                    reply = {"ok": True, "result": await handle(msg)}
                except Exception as e:  # report and keep the connection
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                await socket.send_bytes(protocol.pack(reply))
        except WebSocketDisconnect:
            pass
        finally:
            server.clients -= 1

    return app
