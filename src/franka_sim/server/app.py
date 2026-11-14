"""``RobotServer``: run a ``SimRobot`` in real time, show it in MuJoCo's viewer, serve the model API.

Threads: MuJoCo stepping, the native viewer and OpenGL rendering all stay on the thread that calls
``serve_forever`` (it must be the main thread on macOS). uvicorn serves the API on a background
thread; every request that touches the simulation is queued as a job and run between control
ticks, so the real-time loop and the network never race.

Endpoints:

* ``WS /ws``: the model API (msgpack, see ``protocol.py`` and ``franka_sim.client.RobotClient``).
* ``GET /api/state``: the robot and episode status as JSON (for health checks and dashboards).
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from concurrent.futures import Future

from franka_sim._signals import interruptible
from franka_sim.server import protocol
from franka_sim.server.robot import SimRobot


class RobotServer:
    """Real-time host for a ``SimRobot``.

    Args:
        robot: the simulated robot.
        host / port: where the model API listens.
        realtime: start with the clock running (``False``: paused; the robot advances only on
            ``step`` requests, i.e. lockstep).
        viewer: open MuJoCo's native viewer on the robot's world (CPU backend). On macOS this needs
            ``mjpython``. You can perturb objects in it: double-click a body to select it, then
            Ctrl + right-drag to push it (Ctrl + left-drag to rotate).
    """

    def __init__(self, robot: SimRobot, host: str = "127.0.0.1", port: int = 8000, *, realtime: bool = True,
                 viewer: bool = True):
        self.robot = robot
        self.host, self.port = host, port
        self.running = realtime
        self.use_viewer = viewer
        self._viewer = None
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._state: dict = {}
        self.clients = 0
        self.rtf = 0.0  # measured real-time factor
        self._uvicorn = None

    # Cross-thread calls --------------------------------------------------------------------------
    def call(self, fn, *args, **kwargs) -> Future:
        """Run ``fn`` on the simulation thread; returns a ``Future`` with its result."""
        fut: Future = Future()
        self._jobs.put((fn, args, kwargs, fut))
        return fut

    async def acall(self, fn, *args, **kwargs):
        return await asyncio.wrap_future(self.call(fn, *args, **kwargs))

    def _run_jobs(self, timeout: float) -> None:
        try:
            job = self._jobs.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return
        while True:
            fn, args, kwargs, fut = job
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args, **kwargs))
                except Exception as e:  # report to the caller, keep serving
                    fut.set_exception(e)
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return

    def _snapshot_state(self) -> None:
        s = self.robot.state()
        s.update(running=self.running, rtf=round(self.rtf, 3), clients=self.clients)
        self._state = s

    def state(self) -> dict:
        """The latest state snapshot (safe to read from any thread)."""
        return self._state

    # Control (run on the simulation thread through ``call``) -------------------------------------
    def set_running(self, running: bool) -> dict:
        self.running = running
        self._snapshot_state()
        return self._state

    def lockstep(self, action, mode: str, cameras=(), image_size=None, privileged=False, source="api") -> dict:
        if self.running:
            raise RuntimeError("the robot is running in real time; pause it before stepping (op 'pause')")
        self.robot.step(action, mode, source)
        self._snapshot_state()
        self._sync_viewer()
        return self.robot.transition(cameras, image_size, privileged)

    def reset(self, seed=None, options=None, task=None, cameras=(), image_size=None, privileged=False) -> dict:
        old_model = self.robot.env.model
        self.robot.reset(seed=seed, options=options, task=task)
        if self._viewer is not None and self.robot.env.model is not old_model:  # new task: new world
            self._close_viewer()
            self._open_viewer()
        self._snapshot_state()
        self._sync_viewer()
        return {"obs": self.robot.observation(cameras, image_size, privileged),
                "info": self.robot.transition()["info"]}

    # Native viewer -----------------------------------------------------------------------------
    def _open_viewer(self) -> None:
        import mujoco.viewer

        env = self.robot.env
        if not hasattr(env.sim, "datas"):
            raise RuntimeError("the native viewer needs the CPU backend (or pass viewer=False)")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                self._viewer = mujoco.viewer.launch_passive(env.model, env.sim.datas[0])
                return
            except RuntimeError as e:
                if sys.platform == "darwin" and "mjpython" in str(e):
                    raise RuntimeError("on macOS the MuJoCo viewer needs mjpython: run "
                                       "`uv run --extra server mjpython scripts/serve.py ...`, "
                                       "or pass --headless") from None
                # Closing a passive viewer completes asynchronously (e.g. after a task switch).
                if "already open" not in str(e) or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def _close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():  # the window was closed: stop serving
            self._viewer = None
            self.shutdown()
            return
        self._viewer.sync()

    # Main loop ------------------------------------------------------------------------------------
    def serve_forever(self) -> None:
        """Start the API server and run the simulation loop on this thread until ``shutdown``
        (Ctrl+C, or closing the viewer window)."""
        import uvicorn

        from franka_sim.server.web import create_app

        if self.use_viewer:
            self._open_viewer()
        config = uvicorn.Config(create_app(self), host=self.host, port=self.port, log_level="warning",
                                ws_max_size=64 << 20)
        self._uvicorn = uvicorn.Server(config)
        web = threading.Thread(target=self._uvicorn.run, name="franka-sim-api", daemon=True)
        web.start()
        self._snapshot_state()

        dt = self.robot.control_dt
        now = time.perf_counter()
        next_tick, next_state, next_view = now, now, now
        window_start, window_ticks = now, 0
        sigint = interruptible()  # Ctrl+C stops the server, also under mjpython (which ignores SIGINT)
        sigint.__enter__()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                due = min(next_state, next_view, next_tick if self.running else next_state)
                self._run_jobs(min(due - now, 0.05))
                now = time.perf_counter()
                if self.running and now >= next_tick:
                    self.robot.tick()
                    window_ticks += 1
                    next_tick += dt
                    if now - next_tick > 0.25:  # fell far behind (slow host): do not try to catch up
                        next_tick = now + dt
                    self._sync_viewer()
                    next_view = now + 1 / 30
                elif not self.running:
                    next_tick = now
                if now >= next_view:  # keep the viewer responsive while paused
                    self._sync_viewer()
                    next_view = now + 1 / 30
                if now >= next_state:
                    if now - window_start >= 1.0:
                        self.rtf = window_ticks * dt / (now - window_start)
                        window_start, window_ticks = now, 0
                    self._snapshot_state()
                    next_state = now + 0.05
        except KeyboardInterrupt:
            pass
        finally:
            sigint.__exit__(None, None, None)
            self._uvicorn.should_exit = True
            web.join(timeout=5)
            self._close_viewer()
            self.robot.close()

    def shutdown(self) -> None:
        self._stop.set()


def serve(task: str = "hammer_nails", host: str = "127.0.0.1", port: int = 8000, **kwargs) -> None:
    """Build a ``SimRobot`` and serve it until interrupted (see ``scripts/serve.py``)."""
    server_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in ("realtime", "viewer")}
    robot = SimRobot(task, **kwargs)
    RobotServer(robot, host, port, **server_kwargs).serve_forever()


__all__ = ["RobotServer", "serve", "protocol"]
