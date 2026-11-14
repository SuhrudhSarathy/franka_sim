"""Keyboard teleop through a small console window (pygame, in a child process).

The console shows the operator the robot's cameras, who is in control, the instruction and the
episode status, and reads the keyboard while it has focus:

======================  =============================================
W / S                   +x / -x  (away from / toward the robot base)
A / D                   +y / -y  (robot's left / right)
R / F                   up / down
Q / E                   yaw left / right
I / K, J / L            pitch, roll
Shift (held)            precision mode (1/4 speed)
Space                   toggle the gripper
Tab or Return           take control / hand back to the learner
P                       pause / resume
N                       end the episode and save it
Backspace               discard the episode
[ / ]                   slower / faster
Esc or close window     quit
======================  =============================================

Why a separate process: MuJoCo's viewer only reports key presses (not releases), so it cannot
tell a held key from a tap; and on macOS every GUI window must live on the main thread of its
process, which the MuJoCo viewer occupies under ``mjpython``. The console is started as
``python -m franka_sim.teleop.keyboard`` (not ``multiprocessing``, which would re-import the
caller's main script). Frames and the held-key twist share one memory block (latest frame wins,
never blocks the simulation); events come back as lines on the child's stdout and status lines go
to its stdin, so the console exits by itself if the simulation dies. Needs the ``teleop`` extra.
"""

from __future__ import annotations

import os
import queue
from multiprocessing import shared_memory

import numpy as np

from franka_sim.teleop.base import TeleopDevice, TeleopInput

# key name -> (twist axis, sign)
MOTION_KEYS = {
    "w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "r": (2, 1), "f": (2, -1),
    "l": (3, 1), "j": (3, -1), "i": (4, 1), "k": (4, -1), "q": (5, 1), "e": (5, -1),
}
EVENT_KEYS = {
    "space": "grip", "tab": "toggle", "return": "toggle", "p": "pause", "n": "save",
    "backspace": "discard", "escape": "quit", "[": "slower", "]": "faster",
}
LEGEND = ("WASD move   R/F up/down   Q/E yaw   I/K pitch   J/L roll   Shift fine   Space gripper",
          "Tab take over / hand back   P pause   N save   Backspace discard   [ ] speed   Esc quit")


def keys_to_twist(pressed: set[str]) -> np.ndarray:
    """Twist (6,) from the set of held key names."""
    twist = np.zeros(6)
    for key in pressed:
        if key in MOTION_KEYS:
            axis, sign = MOTION_KEYS[key]
            twist[axis] += sign
    return np.clip(twist, -1.0, 1.0)


_HEADER = 64  # bytes: frame sequence number (uint64) + twist (7 x float64: 6 axes + fine flag)


def _views(buf, n: int, h: int, w: int):
    seq = np.ndarray((1,), dtype=np.uint64, buffer=buf, offset=0)
    twist = np.ndarray((7,), dtype=np.float64, buffer=buf, offset=8)
    frames = np.ndarray((max(n, 1), h, w, 3), dtype=np.uint8, buffer=buf, offset=_HEADER)
    return seq, twist, frames


def _console(shm_name: str, n: int, h: int, w: int, title: str) -> None:
    """Child process: the pygame window."""
    import json
    import sys
    import threading

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    shm = shared_memory.SharedMemory(name=shm_name)
    try:  # Python < 3.13 registers attached blocks for deletion at exit; the simulation owns this one
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    seq, twist, frames = _views(shm.buf, n, h, w)
    latest = {"hud": {}}
    stop = threading.Event()

    def read_stdin():  # status lines from the simulation; EOF = the simulation is gone
        for line in sys.stdin:
            if line.strip() == "stop":
                break
            try:
                latest["hud"] = json.loads(line)
            except ValueError:
                pass
        stop.set()

    threading.Thread(target=read_stdin, daemon=True).start()

    def emit(event: str) -> None:
        try:
            sys.stdout.write(event + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            stop.set()

    pygame.init()
    panel_h = 150
    screen = pygame.display.set_mode((max(n, 1) * w, h + panel_h))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("Menlo,Monaco,Consolas,monospace", 15)
    big = pygame.font.SysFont("Helvetica,Arial", 26, bold=True)
    small = pygame.font.SysFont("Menlo,Monaco,Consolas,monospace", 12)
    clock = pygame.time.Clock()
    seen, surfaces = None, []
    colors = {"HUMAN": (240, 150, 40), "LEARNER": (70, 150, 255), "PAUSED": (140, 140, 150)}
    codes = {name: pygame.key.key_code(name) for name in MOTION_KEYS}
    snapshot, frame_count = os.environ.get("FRANKA_SIM_CONSOLE_SNAPSHOT"), 0
    try:
        while not stop.is_set():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    emit("quit")
                elif ev.type == pygame.KEYDOWN:
                    name = pygame.key.name(ev.key)
                    if name in EVENT_KEYS:
                        emit(EVENT_KEYS[name])
            focused = pygame.key.get_focused()
            pressed = pygame.key.get_pressed()
            held = {name for name, code in codes.items() if focused and pressed[code]}
            twist[:6] = keys_to_twist(held)
            twist[6] = 1.0 if focused and pygame.key.get_mods() & pygame.KMOD_SHIFT else 0.0

            if n > 0 and int(seq[0]) != seen:
                seen = int(seq[0])
                surfaces = [pygame.image.frombuffer(np.ascontiguousarray(frames[i]).tobytes(), (w, h), "RGB")
                            for i in range(n)]
            hud = latest["hud"]
            screen.fill((18, 20, 26))
            names = hud.get("cameras", [])
            for i, surf in enumerate(surfaces):
                screen.blit(surf, (i * w, 0))
                if i < len(names):
                    screen.blit(small.render(names[i], True, (235, 235, 235)), (i * w + 6, h - 18))
            mode = hud.get("mode", "waiting for the simulation")
            color = colors.get(mode, (200, 200, 200))
            pygame.draw.rect(screen, color, (0, h, screen.get_width(), 6))
            screen.blit(big.render(mode, True, color), (12, h + 14))
            screen.blit(font.render(hud.get("status", ""), True, (225, 225, 230)), (170, h + 20))
            y = h + 54
            for line in (hud.get("instruction", ""), hud.get("detail", "")):
                screen.blit(font.render(line, True, (200, 205, 215)), (12, y))
                y += 22
            for k, part in enumerate(LEGEND):
                screen.blit(small.render(part, True, (130, 135, 150)), (12, h + panel_h - 36 + 17 * k))
            if not focused:
                warn = small.render("click this window to drive the robot", True, (240, 150, 40))
                screen.blit(warn, (screen.get_width() - warn.get_width() - 10, h + 16))
            pygame.display.flip()
            frame_count += 1
            if snapshot and frame_count == 45:  # debugging aid: FRANKA_SIM_CONSOLE_SNAPSHOT=path.png
                pygame.image.save(screen, snapshot)
            clock.tick(60)
    finally:
        twist[:] = 0.0
        pygame.quit()
        del seq, twist, frames
        shm.close()


class KeyboardTeleop(TeleopDevice):
    """Keyboard teleop console (see the module docstring).

    Args:
        num_frames: camera frames shown side by side.
        frame_size: ``(height, width)`` of each frame.
        title: window title.
    """

    def __init__(self, num_frames: int = 2, frame_size: tuple[int, int] = (300, 400),
                 title: str = "franka-sim teleop"):
        import subprocess
        import sys
        import threading

        self.n = num_frames
        self.h, self.w = frame_size
        size = _HEADER + max(self.n, 1) * self.h * self.w * 3
        self._shm = shared_memory.SharedMemory(create=True, size=size)
        self._seq, self._twist, self._frames = _views(self._shm.buf, self.n, self.h, self.w)
        self._twist[:] = 0.0
        self._seq[0] = 0
        self._events: queue.Queue = queue.Queue()
        self._hud: queue.Queue = queue.Queue(maxsize=2)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "franka_sim.teleop.keyboard", self._shm.name, str(self.n), str(self.h),
             str(self.w), title],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read_events, daemon=True).start()
        threading.Thread(target=self._write_hud, daemon=True).start()

    def _read_events(self) -> None:
        for line in self._proc.stdout:
            if line.strip():
                self._events.put(line.strip())

    def _write_hud(self) -> None:
        import json

        while True:
            hud = self._hud.get()
            try:
                self._proc.stdin.write((hud if isinstance(hud, str) else json.dumps(hud)) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return
            if hud == "stop":
                return

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def poll(self) -> TeleopInput:
        events = []
        try:
            while True:
                events.append(self._events.get_nowait())
        except queue.Empty:
            pass
        if not self.alive:
            events.append("quit")
        vals = self._twist.copy()
        return TeleopInput(twist=vals[:6], events=events, fine=bool(vals[6] > 0.5))

    def show(self, frames: list[np.ndarray], hud: dict) -> None:
        for i, img in enumerate(frames[: self.n]):
            if img.shape[:2] != (self.h, self.w):
                raise ValueError(f"frames must be {(self.h, self.w)}, got {img.shape[:2]}")
            self._frames[i] = img
        self._seq[0] += 1
        try:
            self._hud.put_nowait(hud)
        except queue.Full:  # the console is behind: drop this status line, the next one replaces it
            pass

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(queue.Full):
            self._hud.put("stop", timeout=0.5)
        try:
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        del self._seq, self._twist, self._frames
        self._shm.close()
        self._shm.unlink()


if __name__ == "__main__":  # the console process
    import sys

    _console(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
