"""Parametric rigid-object proxies; dimensions/masses are assumptions, not scans.

Screw's shaft is horizontal. Pliers are closed and rigid. Ring-like objects use
multiple convex colliders so their holes are not filled by a convex hull.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np

from ..scene import Box, ObjectPart as Part

OBJECT_KINDS = ("cube", "flat", "triangle", "cylinder", "pliers", "screw",
                "nut", "washer", "l_bracket", "t_bracket", "capsule", "wedge")


@dataclass(frozen=True)
class ObjectSpec:
    kind: str
    half_size: tuple[float, float, float]
    mass: float
    parts: tuple[Part, ...]
    yaw_period: float = 2 * math.pi

    def box(self, name="cube"):
        colors = {"screw": (.55,.57,.62,1.), "pliers": (.28,.31,.36,1.),
                  "washer": (.65,.67,.72,1.), "nut": (.6,.62,.68,1.),
                  "cylinder": (.2,.65,.35,1.), "flat": (.15,.4,.8,1.),
                  "triangle": (.9,.65,.1,1.), "capsule": (.65,.25,.7,1.)}
        return Box(name, self.half_size, (.2, 0.), mass=self.mass, parts=self.parts,
                   rgba=colors.get(self.kind,(.85,.25,.2,1.)))


def _ring(segments, radius, width, height):
    parts = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        # Tangential boxes form a closed polygon around a real central hole.
        parts.append(Part("box", (radius * math.tan(math.pi / segments), width, height),
                          (radius * math.cos(a), radius * math.sin(a), 0.),
                          (math.cos((a+math.pi/2)/2), 0., 0., math.sin((a+math.pi/2)/2))))
    return tuple(parts)


@lru_cache(maxsize=128)
def object_spec(kind="cube", scale=1.0):
    if not .5 <= scale <= 1.5:
        raise ValueError("Object scale must be in [0.5, 1.5]")
    horizontal = (math.sqrt(.5), 0., math.sqrt(.5), 0.)
    triangle = tuple((x,y,z) for z in (-.009,.009) for x,y in ((-.022,-.018),(.022,-.018),(0.,.024)))
    wedge = ((-.025,-.016,-.01),(.025,-.016,-.01),(-.025,-.016,.01),
             (-.025,.016,-.01),(.025,.016,-.01),(-.025,.016,.01))
    presets = {
        "cube": ObjectSpec("cube", (.015,.015,.015), .03, (Part("box",(.015,.015,.015)),), math.pi/2),
        "flat": ObjectSpec("flat", (.026,.018,.004), .025, (Part("box",(.026,.018,.004)),), math.pi),
        "triangle": ObjectSpec("triangle", (.022,.024,.009), .025,
                               (Part("mesh",(1.,1.,1.), vertices=triangle),)),
        "cylinder": ObjectSpec("cylinder", (.013,.013,.024), .035,
                               (Part("cylinder",(.013,.024,0.)),), math.pi/2),
        "pliers": ObjectSpec("pliers", (.065,.022,.008), .11, (
            Part("box",(.020,.011,.008),(.042,0.,0.),mass_fraction=3.),
            Part("capsule",(.007,.035,0.),(-.020,-.014,0.),horizontal),
            Part("capsule",(.007,.035,0.),(-.020,.014,0.),horizontal),
            Part("box",(.009,.021,.007),(.014,0.,0.),mass_fraction=2.))),
        "screw": ObjectSpec("screw", (.029,.008,.008), .018, (
            Part("cylinder",(.004,.023,0.),(0.,0.,0.),horizontal,mass_fraction=2.),
            Part("cylinder",(.008,.003,0.),(-.026,0.,0.),horizontal))),
        "nut": ObjectSpec("nut", (.017,.017,.006), .025, _ring(6,.011,.004,.006)),
        "washer": ObjectSpec("washer", (.022,.022,.0025), .012, _ring(10,.016,.004,.0025)),
        "l_bracket": ObjectSpec("l_bracket", (.027,.027,.006), .035, (
            Part("box",(.027,.007,.006),(0.,-.020,0.)),
            Part("box",(.007,.020,.006),(-.020,.007,0.)))),
        "t_bracket": ObjectSpec("t_bracket", (.025,.030,.007), .035, (
            Part("box",(.025,.008,.007),(0.,.022,0.)),
            Part("box",(.008,.022,.007),(0.,-.008,0.)))),
        "capsule": ObjectSpec("capsule", (.034,.012,.012), .03,
                              (Part("capsule",(.012,.022,0.),quat=horizontal),), math.pi),
        "wedge": ObjectSpec("wedge", (.025,.016,.010), .03,
                            (Part("mesh",(1.,1.,1.),vertices=wedge),)),
    }
    if kind not in presets:
        raise ValueError(f"Unknown object {kind!r}; choose from {OBJECT_KINDS}")
    obj = presets[kind]
    parts = tuple(replace(p, size=tuple(x*scale for x in p.size) if p.kind != "mesh" else p.size,
                          pos=tuple(x*scale for x in p.pos),
                          vertices=tuple(tuple(x*scale for x in v) for v in p.vertices)) for p in obj.parts)
    return replace(obj, half_size=tuple(x*scale for x in obj.half_size),
                   mass=obj.mass*scale**3, parts=parts)


def descriptor(kind, scale=1.0):
    obj = object_spec(kind, scale)
    onehot = np.zeros(len(OBJECT_KINDS), np.float32)
    onehot[OBJECT_KINDS.index(kind)] = 1.
    return np.r_[onehot, np.asarray(obj.half_size, np.float32) / .05].astype(np.float32)


@lru_cache(maxsize=128)
def support_parts(kind, scale):
    """Part rotations for exact support of primitive or convex mesh geometries."""
    out = []
    for part in object_spec(kind, scale).parts:
        w,x,y,z = part.quat
        R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        out.append((part, R))
    return tuple(out)


def bottom_height(xp, task, pos, rot):
    """Height of the lowest point, not just centre minus the initial thickness.

    This prevents an upright long object touching the table from counting as lifted.
    """
    if not task.shape_conditioned:
        return pos[..., 2] - task.cube_half
    from .task import _as
    lowest = None
    for part, R in support_parts(task.object_kind, task.object_scale):
        row = rot[..., 2, :] @ _as(xp, R, pos)
        centre = (rot[..., 2, :] * _as(xp, part.pos, pos)).sum(-1)
        if part.kind == "box":
            support = (abs(row) * _as(xp, part.size, pos)).sum(-1)
            value = centre - support
        elif part.kind == "cylinder":
            value = centre - part.size[0] * (row[..., 0]**2 + row[..., 1]**2)**.5 - part.size[1]*abs(row[..., 2])
        elif part.kind == "capsule":
            value = centre - part.size[0] - part.size[1]*abs(row[..., 2])
        else:
            vertices = _as(xp, np.asarray(part.vertices).T, pos)
            dots = row @ vertices
            value = centre + (dots.min(-1) if xp is np else dots.min(-1).values)
        lowest = value if lowest is None else xp.minimum(lowest, value)
    return pos[..., 2] + lowest
