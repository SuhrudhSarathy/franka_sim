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


def add_cell(body: mujoco.MjsBody, prefix: str, center, inner_half, height: float, wall: float = WALL,
             floor: float = 0.0, rgba=GRAY, floor_rgba=GRAY) -> None:
    """Add an open-topped rectangular cell (4 walls, optional floor) to ``body``.

    ``center`` is the (x, y) of the cell in ``body``'s frame, ``inner_half`` the half size of the
    opening, ``height`` the wall height above the body origin, ``floor`` the floor thickness.
    """
    cx, cy = center
    ix, iy = inner_half
    ox, oy = ix + wall, iy + wall
    if floor > 0:
        body.add_geom(name=f"{prefix}_floor", type=BOX, size=[ox, oy, floor / 2], pos=[cx, cy, floor / 2],
                      rgba=floor_rgba)
    hz = (height - floor) / 2
    z = floor + hz
    for j, (dx, dy, sx, sy) in enumerate([
        (ix + wall / 2, 0, wall / 2, oy), (-(ix + wall / 2), 0, wall / 2, oy),
        (0, iy + wall / 2, ix, wall / 2), (0, -(iy + wall / 2), ix, wall / 2),
    ]):
        body.add_geom(name=f"{prefix}_wall{j}", type=BOX, size=[sx, sy, hz], pos=[cx + dx, cy + dy, z], rgba=rgba)


def grasp_offset_for_height(half_height: float) -> float:
    """Grasp point (EE site height above the object's centre) for an object of this half height
    resting on the table: as high as possible while the finger pads still overlap the object, but
    with the fingertips (12 mm below the site) clear of the table."""
    return max(0.013, 2 * half_height - 0.006) - half_height


def _goal_site(body: mujoco.MjsBody, name: str, pos) -> None:
    body.add_site(name=name, pos=list(pos), size=[0.004, 0, 0], rgba=[0, 0, 0, 0])


def add_receptacle(spec: mujoco.MjSpec, colors, name: str = "receptacle", spacing: float = 0.11,
                   pos=(0.62, 0.0, 0.0), peg_half=PEG_HALF) -> mujoco.MjsBody:
    """A mocap block with one square hole per color, in a row along its local y.

    Each hole is four walls plus a floor; the rim is tinted with the hole's color. Site ``hole{i}``
    marks the bottom centre of hole ``i``; ``goal{i}`` where peg ``i``'s centre ends up when seated.
    """
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    inner = peg_half[0] + HOLE_CLEARANCE  # half width of the opening
    floor_h = 0.005
    for i, color in enumerate(colors):
        y = (i - (len(colors) - 1) / 2) * spacing
        tint = tuple(0.5 * np.array(color[:3]) + 0.25) + (1.0,)
        add_cell(body, f"hole{i}", (0, y), (inner, inner), RECEPTACLE_HEIGHT, floor=floor_h, rgba=tint)
        _goal_site(body, f"hole{i}", [0, y, floor_h])
        _goal_site(body, f"goal{i}", [0, y, floor_h + peg_half[2]])
    return body


# --------------------------------------------------------------------------------------------
# Gears
# --------------------------------------------------------------------------------------------
GEAR_THICKNESS = 0.012
SHAFT_RADIUS = 0.0075
GEAR_BORE = SHAFT_RADIUS + 0.003  # radius of the bore (clearance 3 mm)
GEAR_PLATE_HALF_Z = 0.008
SHAFT_HEIGHT = 0.045  # above the plate
STEEL = (0.55, 0.57, 0.6, 1.0)
GEAR_COLORS = ((0.85, 0.6, 0.2, 1.0), (0.7, 0.72, 0.75, 1.0), (0.45, 0.5, 0.58, 1.0))  # brass, steel, dark


def add_gear(spec: mujoco.MjSpec, name: str, radius: float, color, pos, teeth: int | None = None,
             segments: int = 10) -> mujoco.MjsBody:
    """A spur gear lying flat: a ring of box segments around a bore (collision) plus visual teeth.

    ``radius`` is the pitch radius. The collision ring spans ``GEAR_BORE`` .. ``radius - 0.002``, so
    gears placed on shafts ``r_a + r_b`` apart look meshed without their teeth colliding. Body
    origin = centre of the gear (mid-thickness).
    """
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    h = GEAR_THICKNESS / 2
    r_in, r_out = GEAR_BORE, radius - 0.002
    rm = (r_in + r_out) / 2
    seg_half_t = rm * np.pi / segments * 1.08
    mass = 0.02 + 40.0 * radius ** 2
    # Collision bitmasks: the bore ring segments (bit 4) touch only the shafts; a smooth full-radius
    # disc (bit 1, hidden) is what the fingers, table, plate and other gears touch, and it ignores the
    # shafts so it does not block the bore. Gripping a smooth disc avoids the polygon's slanted facets
    # squeezing the gear out of the fingers, and keeps contact counts low for MuJoCo Warp.
    for k in range(segments):
        a = 2 * np.pi * k / segments
        body.add_geom(name=f"{name}_ring{k}", type=BOX, size=[(r_out - 0.001 - r_in) / 2, seg_half_t, h],
                      pos=[(r_in + r_out - 0.001) / 2 * np.cos(a), (r_in + r_out - 0.001) / 2 * np.sin(a), 0],
                      quat=_zquat(a), rgba=color, mass=mass / segments, contype=4, conaffinity=4)
    body.add_geom(name=f"{name}_grip", type=CYL, size=[r_out, h, 0], rgba=color, group=3,
                  contype=1, conaffinity=1, mass=0.0001, **GRIP)
    teeth = teeth or max(8, int(round(radius * 600)))
    for k in range(teeth):
        a = 2 * np.pi * (k + 0.5) / teeth
        body.add_geom(name=f"{name}_tooth{k}", type=BOX, size=[0.0035, np.pi * radius / teeth * 0.5, h * 0.95],
                      pos=[(radius + 0.0005) * np.cos(a), (radius + 0.0005) * np.sin(a), 0], quat=_zquat(a),
                      rgba=color, contype=0, conaffinity=0, mass=0.0001)
    # Hub ring marking (visual).
    body.add_geom(name=f"{name}_hub", type=CYL, size=[r_in + 0.004, h + 0.0005, 0], rgba=(0.2, 0.2, 0.22, 1),
                  contype=0, conaffinity=0, mass=0.0001)
    return body


def _zquat(a: float) -> list[float]:
    return [float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))]


def add_gear_plate(spec: mujoco.MjSpec, radii, name: str = "gear_plate", pos=(0.6, 0.0, GEAR_PLATE_HALF_Z)):
    """A mocap base plate with one vertical shaft per gear; neighbouring shafts are spaced by the
    sum of the gears' pitch radii (so assembled gears mesh). ``goal{i}`` = seated centre of gear i."""
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    ys = [0.0]
    for a, b in zip(radii[:-1], radii[1:]):
        ys.append(ys[-1] + a + b + 0.001)
    ys = np.array(ys) - (ys[0] + ys[-1]) / 2
    half_y = (ys[-1] - ys[0]) / 2 + max(radii) + 0.02
    body.add_geom(name=f"{name}_geom", type=BOX, size=[max(radii) + 0.03, half_y, GEAR_PLATE_HALF_Z], rgba=GRAY)
    top = GEAR_PLATE_HALF_Z
    for i, y in enumerate(ys):
        # Capsule shafts: the rounded tip guides the bore on.
        body.add_geom(name=f"shaft{i}", type=CAPSULE, size=[SHAFT_RADIUS, SHAFT_HEIGHT / 2 - SHAFT_RADIUS, 0],
                      pos=[0, y, top + SHAFT_HEIGHT / 2], rgba=STEEL, contype=4, conaffinity=4)
        body.add_geom(name=f"shaft{i}_collar", type=CYL, size=[radii[i] * 0.6, 0.001, 0],
                      pos=[0, y, top + 0.001], rgba=GEAR_COLORS[i % 3], contype=0, conaffinity=0)
        _goal_site(body, f"shaft{i}", [0, y, top])
        _goal_site(body, f"goal{i}", [0, y, top + GEAR_THICKNESS / 2])
    return body


# --------------------------------------------------------------------------------------------
# Kitting
# --------------------------------------------------------------------------------------------
# name -> (geom type, size, rgba). Sizes are MuJoCo geom sizes (half extents / radius, half length).
KIT_PARTS = {
    "block": (BOX, (0.026, 0.016, 0.016), (0.85, 0.45, 0.1, 1.0)),
    "cylinder": (CYL, (0.017, 0.022, 0), (0.6, 0.62, 0.66, 1.0)),
    "bar": (BOX, (0.045, 0.011, 0.011), (0.2, 0.4, 0.75, 1.0)),
    "cube": (BOX, (0.016, 0.016, 0.016), (0.15, 0.6, 0.3, 1.0)),
}
KIT_CLEARANCE = 0.012
KIT_WALL_HEIGHT = 0.03


def kit_part_footprint(name: str) -> tuple[float, float, float]:
    """(half x, half y, half height) of a kit part."""
    gtype, size, _ = KIT_PARTS[name]
    if gtype == CYL:
        return size[0], size[0], size[1]
    return size[0], size[1], size[2]


def add_kit_part(spec: mujoco.MjSpec, name: str, kind: str, pos) -> mujoco.MjsBody:
    gtype, size, rgba = KIT_PARTS[kind]
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(name=f"{name}_geom", type=gtype, size=list(size), rgba=rgba, mass=0.04, **GRIP)
    return body


def add_kit_tray(spec: mujoco.MjSpec, kinds, name: str = "kit_tray", pos=(0.6, 0.0, 0.0)) -> mujoco.MjsBody:
    """A mocap tray with one fitted compartment per part, side by side along the tray's local y.

    Each compartment is sized to its part plus ``KIT_CLEARANCE`` and has a faint outline of the
    part's colour on its floor. ``goal{i}`` = centre of part ``i`` when it sits in its compartment.
    """
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    floor = 0.004
    cells = [kit_part_footprint(k) for k in kinds]
    cell_x = max(c[0] for c in cells) + KIT_CLEARANCE + WALL / 2
    cell_y = max(c[1] for c in cells) + KIT_CLEARANCE + WALL / 2
    ncol = len(kinds)
    nrow = 1
    for i, kind in enumerate(kinds):
        r, c = divmod(i, ncol)
        cx = (r - (nrow - 1) / 2) * 2 * (cell_x + WALL / 2)
        cy = ((ncol - 1) / 2 - c) * 2 * (cell_y + WALL / 2)
        hx, hy, hz = cells[i]
        rgba = KIT_PARTS[kind][2]
        add_cell(body, f"kit{i}", (cx, cy), (cell_x - WALL / 2, cell_y - WALL / 2), KIT_WALL_HEIGHT,
                 wall=WALL, floor=floor, rgba=(0.3, 0.32, 0.36, 1.0), floor_rgba=(0.22, 0.23, 0.26, 1.0))
        body.add_geom(name=f"kit{i}_outline", type=BOX, size=[hx + 0.003, hy + 0.003, 0.0006],
                      pos=[cx, cy, floor + 0.0006], rgba=tuple(0.6 * np.array(rgba[:3])) + (1.0,),
                      contype=0, conaffinity=0)
        _goal_site(body, f"goal{i}", [cx, cy, floor + hz])
    return body


# --------------------------------------------------------------------------------------------
# PCB (card-edge connector) insertion
# --------------------------------------------------------------------------------------------
PCB_HALF = (0.035, 0.0018, 0.024)  # length, thickness, height
SLOT_CLEARANCE = 0.0014  # per side
MOTHERBOARD_HALF_Z = 0.006
CONNECTOR_HEIGHT = 0.012
PCB_GREEN = (0.05, 0.4, 0.15, 1.0)
GOLD = (0.85, 0.7, 0.25, 1.0)


def add_pcb(spec: mujoco.MjSpec, name: str = "pcb", pos=(0.4, 0.0, PCB_HALF[2])) -> mujoco.MjsBody:
    """An expansion card standing on its edge: green board with chips and gold edge contacts."""
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    hx, hy, hz = PCB_HALF
    body.add_geom(name=f"{name}_board", type=BOX, size=list(PCB_HALF), rgba=PCB_GREEN, mass=0.03, **GRIP)
    body.add_geom(name=f"{name}_contacts", type=BOX, size=[hx * 0.85, hy + 0.0002, 0.003],
                  pos=[0, 0, -hz + 0.003], rgba=GOLD, contype=0, conaffinity=0, mass=0.0001)
    for k, (x, z, sx, sz) in enumerate([(-0.015, 0.006, 0.008, 0.007), (0.013, 0.004, 0.011, 0.006)]):
        for side in (1, -1):
            body.add_geom(name=f"{name}_chip{k}_{side}", type=BOX, size=[sx, 0.0008, sz],
                          pos=[x, side * (hy + 0.0008), z], rgba=(0.08, 0.08, 0.09, 1.0),
                          contype=0, conaffinity=0, mass=0.0001)
    return body


def add_card_rack(spec: mujoco.MjSpec, name: str = "card_rack", pos=(0.4, -0.25, 0.0)) -> mujoco.MjsBody:
    """A mocap stand with a slot that holds the card upright (card bottom 1 cm into the slot)."""
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    # Generous end clearance so the card's edge does not drag on the slot ends as it is lifted out.
    add_cell(body, f"{name}_slot", (0, 0), (PCB_HALF[0] + 0.012, PCB_HALF[1] + 0.003), 0.012,
             wall=0.008, floor=0.002, rgba=(0.25, 0.25, 0.28, 1.0))
    return body


def add_motherboard(spec: mujoco.MjSpec, num_slots: int = 3, name: str = "motherboard", spacing: float = 0.06,
                    pos=(0.6, 0.0, MOTHERBOARD_HALF_Z)) -> mujoco.MjsBody:
    """A mocap board with ``num_slots`` parallel card-edge connectors (long axis = local x),
    spaced along local y. ``goal{i}`` = centre of a card fully seated in connector ``i``."""
    body = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    half_y = spacing * (num_slots - 1) / 2 + 0.04
    body.add_geom(name=f"{name}_geom", type=BOX, size=[0.07, half_y, MOTHERBOARD_HALF_Z],
                  rgba=(0.1, 0.25, 0.12, 1.0))
    top = MOTHERBOARD_HALF_Z
    inner = (PCB_HALF[0] + 0.004, PCB_HALF[1] + SLOT_CLEARANCE)
    for i in range(num_slots):
        y = ((num_slots - 1) / 2 - i) * spacing  # slot 0 = +y (robot's left)
        cell = body.add_body(name=f"{name}_conn{i}", pos=[0, y, top])
        add_cell(cell, f"conn{i}", (0, 0), inner, CONNECTOR_HEIGHT, wall=0.004, rgba=(0.12, 0.12, 0.13, 1.0))
        _goal_site(body, f"slot{i}", [0, y, top])
        _goal_site(body, f"goal{i}", [0, y, top + PCB_HALF[2]])
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
