"""Accumulate GPU capacity diagnostics across every physics substep."""

import warp as wp


@wp.kernel
def record_capacity(nefc: wp.array(dtype=wp.int32), nacon: wp.array(dtype=wp.int32),
                    overflow: wp.array(dtype=wp.int32), peak_nefc: wp.array(dtype=wp.int32),
                    peak_nacon: wp.array(dtype=wp.int32), seen_overflow: wp.array(dtype=wp.int32)):
    world = wp.tid()
    peak_nefc[world] = wp.max(peak_nefc[world], nefc[world])
    seen_overflow[world] = seen_overflow[world] | overflow[world]
    if world == 0:
        peak_nacon[0] = wp.max(peak_nacon[0], nacon[0])


@wp.kernel
def record_forbidden_contacts(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                              dist: wp.array(dtype=wp.float32), worldid: wp.array(dtype=wp.int32),
                              forbidden: wp.array2d(dtype=wp.int32), unsafe: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i < nacon[0] and dist[i] <= 0.0:
        pair = geom[i]
        if pair[0] >= 0 and pair[1] >= 0 and forbidden[pair[0], pair[1]] != 0:
            wp.atomic_or(unsafe, worldid[i], 1)
