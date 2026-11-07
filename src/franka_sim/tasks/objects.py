"""Procedural assets built from primitive geoms (fast and robust on both MuJoCo and MuJoCo Warp).

Fixtures (nail board, peg receptacle, button panel) are *mocap* bodies: they are immovable, but
their pose lives in ``MjData.mocap_pos`` and so can be randomized independently in every world.
"""

from __future__ import annotations

import mujoco
import numpy as np

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE

COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.95, 0.8, 0.1, 1.0),
    "purple": (0.6, 0.25, 0.75, 1.0),
    "orange": (0.95, 0.5, 0.1, 1.0),
}
COLOR_NAMES = tuple(COLORS)
WOOD = (0.62, 0.45, 0.28, 1.0)
DARK_WOOD = (0.45, 0.3, 0.18, 1.0)
METAL = (0.6, 0.62, 0.65, 1.0)
GRAY = (0.35, 0.35, 0.38, 1.0)

# Grippable objects get torsional friction (condim 4) so they do not pivot between the fingers.
GRIP = {"condim": 4, "friction": [1.5, 0.05, 0.001], "solref": [0.01, 1], "solimp": [0.95, 0.99, 0.001, 0.5, 2]}


# --------------------------------------------------------------------------------------------
# Hammer and nails
# --------------------------------------------------------------------------------------------
HAMMER_HANDLE_HALF = (0.10, 0.013, 0.013)
# Tall head: at the bottom of a strike the handle must still clear neighbouring (undriven) nails.
HAMMER_HEAD_HALF = (0.022, 0.022, 0.07)
HAMMER_HANDLE_Z = HAMMER_HEAD_HALF[2]  # handle height (above the floor) when the hammer rests on its head


def add_hammer(spec: mujoco.MjSpec, name: str = "hammer", pos=(0.45, -0.25, HAMMER_HANDLE_Z)) -> mujoco.MjsBody:
    """A T-shaped hammer: a handle along +x and a head at the +x end extending down (and up).

    The body frame is at the centre of the handle. Sites: ``<name>_grip`` (where to grasp, near the
    centre of mass) and ``<name>_face`` (bottom striking face of the head).
    """
    hx = HAMMER_HANDLE_HALF[0]
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(name=f"{name}_handle", type=BOX, size=list(HAMMER_HANDLE_HALF), rgba=WOOD, mass=0.08, **GRIP)
    head_pos = [hx + HAMMER_HEAD_HALF[0], 0, 0]
    body.add_geom(name=f"{name}_head", type=BOX, size=list(HAMMER_HEAD_HALF), pos=head_pos, rgba=METAL,
                  mass=0.15, friction=[0.8, 0.005, 0.0001])
    body.add_site(name=f"{name}_grip", pos=[0.05, 0, 0], size=[0.005, 0, 0], rgba=[0, 0, 0, 0])
    body.add_site(name=f"{name}_face", pos=[head_pos[0], 0, -HAMMER_HEAD_HALF[2]], size=[0.005, 0, 0],
                  rgba=[0, 0, 0, 0])
    return body


def add_hammer_rest(spec: mujoco.MjSpec, name: str = "hammer_rest") -> mujoco.MjsBody:
    """A small mocap block that holds the back of the handle level while the hammer rests on its head."""
    h = HAMMER_HANDLE_Z - HAMMER_HANDLE_HALF[2]
    body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.4, -0.25, h / 2])
    body.add_geom(name=f"{name}_geom", type=BOX, size=[0.02, 0.03, h / 2], rgba=GRAY)
    return body


NAIL_SHAFT = (0.005, 0.03)  # radius, half length
NAIL_HEAD = (0.014, 0.004)  # radius, half height
NAIL_TRAVEL = 0.04  # how far a nail can be driven
BOARD_HALF = (0.10, 0.17, 0.02)
NAIL_REGION_HALF = (0.075, 0.145)  # nails are placed within this (x, y) half extent of the board centre
NAIL_HEAD_RGBA = (0.72, 0.74, 0.78, 1.0)


def add_nail_board(spec: mujoco.MjSpec, num_nails: int = 4, name: str = "board",
                   pos=(0.58, 0.07, BOARD_HALF[2]), frictionloss: float = 6.0) -> mujoco.MjsBody:
    """A wooden board with ``num_nails`` identical nails, each on a vertical slide joint.

    The board and every nail socket are separate mocap bodies, so both the board pose and the
    position of each nail on it can be randomized per world (see ``HammerNailsTask.reset``).
    Socket ``nail{i}_base`` sits on the board's top face; nail ``nail{i}`` hangs off it with joint
    ``nail{i}_joint``: 0 when sticking up, ``-NAIL_TRAVEL`` when fully driven in. Joint friction means
    it takes a real push (a hammer blow) to drive a nail.
    """
    board = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    board.add_geom(name=f"{name}_geom", type=BOX, size=list(BOARD_HALF), rgba=DARK_WOOD)
    top = pos[2] + BOARD_HALF[2]
    for i in range(num_nails):
        # Default layout: a row along y (reset() overrides it).
        y = pos[1] + (i - (num_nails - 1) / 2) * 0.08
        base = spec.worldbody.add_body(name=f"nail{i}_base", mocap=True, pos=[pos[0], y, top])
        base.add_geom(name=f"nail{i}_hole", type=CYL, size=[0.009, 0.0008, 0], pos=[0, 0, 0.0008],
                      rgba=(0.2, 0.14, 0.08, 1.0), contype=0, conaffinity=0)
        # gravcomp: soft joint friction would otherwise let the nail creep down under its own weight.
        nail = base.add_body(name=f"nail{i}", pos=[0, 0, NAIL_TRAVEL], gravcomp=1.0)
        nail.add_joint(name=f"nail{i}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                       range=[-NAIL_TRAVEL, 0], damping=2.0, frictionloss=frictionloss, armature=0.01)
        # The shaft is visual only (it passes into the board); the head is what gets hit.
        shaft_len = NAIL_TRAVEL - 0.002 + NAIL_TRAVEL
        nail.add_geom(name=f"nail{i}_shaft", type=CYL, size=[NAIL_SHAFT[0], shaft_len / 2, 0],
                      pos=[0, 0, -shaft_len / 2], rgba=METAL, contype=0, conaffinity=0, mass=0.02)
        nail.add_geom(name=f"nail{i}_head", type=CYL, size=[NAIL_HEAD[0], NAIL_HEAD[1], 0],
                      pos=[0, 0, NAIL_HEAD[1]], rgba=NAIL_HEAD_RGBA, mass=0.01)
        nail.add_site(name=f"nail{i}_top", pos=[0, 0, 2 * NAIL_HEAD[1]], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
    return board


# --------------------------------------------------------------------------------------------
# Pegs and holes
# --------------------------------------------------------------------------------------------
PEG_HALF = (0.016, 0.016, 0.045)
HOLE_CLEARANCE = 0.005
RECEPTACLE_HEIGHT = 0.05
WALL = 0.012


def add_peg(spec: mujoco.MjSpec, name: str, color, pos, half=PEG_HALF) -> mujoco.MjsBody:
    """A square peg standing upright. Site ``<name>_bottom`` marks its bottom face."""
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(name=f"{name}_geom", type=BOX, size=list(half), rgba=color, mass=0.05, **GRIP)
    body.add_site(name=f"{name}_bottom", pos=[0, 0, -half[2]], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
    return body


def add_receptacle(spec: mujoco.MjSpec, colors, name: str = "receptacle", spacing: float = 0.11,
                   pos=(0.62, 0.0, 0.0), peg_half=PEG_HALF) -> mujoco.MjsBody:
    """A mocap block with one square hole per color, in a row along y.

    Each hole is four walls plus a floor; the rim is tinted with the hole's color. Site ``hole{i}``
    marks the bottom centre of hole ``i``.
    """
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    inner = peg_half[0] + HOLE_CLEARANCE  # half width of the opening
    h = RECEPTACLE_HEIGHT
    floor_h = 0.005
    for i, color in enumerate(colors):
        y = (i - (len(colors) - 1) / 2) * spacing
        outer = inner + WALL
        tint = tuple(0.5 * np.array(color[:3]) + 0.25) + (1.0,)
        body.add_geom(name=f"hole{i}_floor", type=BOX, size=[outer, outer, floor_h / 2], pos=[0, y, floor_h / 2],
                      rgba=GRAY)
        wall_z = floor_h + (h - floor_h) / 2
        wall_hz = (h - floor_h) / 2
        for j, (dx, dy, sx, sy) in enumerate([
            (inner + WALL / 2, 0, WALL / 2, outer), (-(inner + WALL / 2), 0, WALL / 2, outer),
            (0, inner + WALL / 2, inner, WALL / 2), (0, -(inner + WALL / 2), inner, WALL / 2),
        ]):
            body.add_geom(name=f"hole{i}_wall{j}", type=BOX, size=[sx, sy, wall_hz], pos=[dx, y + dy, wall_z],
                          rgba=tint)
        body.add_site(name=f"hole{i}", pos=[0, y, floor_h], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
    return body


# --------------------------------------------------------------------------------------------
# Buttons
# --------------------------------------------------------------------------------------------
BUTTON_TRAVEL = 0.012
PANEL_HALF_Z = 0.025


def add_button_panel(spec: mujoco.MjSpec, num_buttons: int = 4, name: str = "panel", spacing: float = 0.09,
                     pos=(0.6, 0.0, PANEL_HALF_Z)) -> mujoco.MjsBody:
    """A mocap panel with spring-loaded push buttons facing up, in a row along y.

    Button ``i`` has joint ``button{i}_joint`` (0 = up, ``-BUTTON_TRAVEL`` = fully pressed).
    """
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    half_y = spacing * (num_buttons - 1) / 2 + 0.05
    body.add_geom(name=f"{name}_geom", type=BOX, size=[0.06, half_y, PANEL_HALF_Z], rgba=GRAY)
    for i in range(num_buttons):
        y = (i - (num_buttons - 1) / 2) * spacing
        color = COLORS[COLOR_NAMES[i % len(COLOR_NAMES)]]
        btn = body.add_body(name=f"button{i}", pos=[0, y, PANEL_HALF_Z])
        btn.add_joint(name=f"button{i}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                      range=[-BUTTON_TRAVEL, 0], stiffness=300.0, springref=0.0, damping=3.0)
        btn.add_geom(name=f"button{i}_cap", type=BOX, size=[0.022, 0.022, 0.008], pos=[0, 0, 0.008],
                     rgba=color, mass=0.02)
        btn.add_site(name=f"button{i}_top", pos=[0, 0, 0.016], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
    return body
