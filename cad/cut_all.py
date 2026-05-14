"""
cad/cut_all.py

Helpers for "no-target cut" — the operation that cuts every workspace body
the tool intersects.  The fan-out logic itself lives in the viewport
(viewer/vp_extrude.py:_do_cut_all_intersecting) because it needs viewport
state to drive CrossBodyCutOp.commit().  Pure helpers go here so they can
be tested without a Qt/OpenGL stack.
"""

from __future__ import annotations


def bboxes_overlap(a, b) -> bool:
    """
    True when two axis-aligned bounding boxes (build123d BoundingBox or
    anything with .min.{X,Y,Z} and .max.{X,Y,Z}) overlap on every axis.

    Used as a cheap prefilter to skip bodies the cut tool can't intersect
    before paying for a boolean operation.
    """
    return (a.min.X <= b.max.X and a.max.X >= b.min.X and
            a.min.Y <= b.max.Y and a.max.Y >= b.min.Y and
            a.min.Z <= b.max.Z and a.max.Z >= b.min.Z)
