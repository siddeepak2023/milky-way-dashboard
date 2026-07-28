"""Pack point clouds into a quantized octree binary (format v3, magic "CSM3").

Why
---
v2 shipped Float32 [x,y,z,size,value] = 20 B/point in one flat array. The whole
layer was uploaded and drawn every frame, so cost scaled with the catalogue, not
with what was on screen. v3 changes both halves of that:

  * 8 B/point instead of 20 B  (int16 x/y/z + uint8 size + uint8 value)
  * points grouped into octree nodes, so the viewer can draw only the nodes that
    are inside the frustum and stop descending once a node is small on screen.

Quantization is per NODE, not per file. A node's positions are stored as int16
offsets inside that node's own cube, so precision is nodeEdge/65535 — for a leaf
node of the star layer that is well under 0.01 pc. A file-wide int16 grid would
have been ~0.06 pc for stars and ~0.24 Mpc for quasars; per-node makes the
quantization error smaller than any structure the viewer can show.

LOD is Potree-style and lossless in aggregate: each node keeps a random subsample
of the points in its subtree (up to CAPACITY) and hands the rest to its children.
Rendering an internal node without its children therefore draws a real random
subsample of that region — density stays proportional to true density, so a
distant view under-samples uniformly rather than inventing or dropping structure.

Format
------
  header (32 B)
    char[4]  "CSM3"
    uint32   version = 3
    uint32   nodeCount
    uint32   pointCount
    float32  vmin, vmax        value (measured scalar) dequantization range
    float32  smin, smax        size/brightness proxy dequantization range
  node table (nodeCount x 64 B)
    float32  bx, by, bz        node cube minimum corner
    float32  edge              node cube edge length
    uint32   offset            first point index of this node
    uint32   count             points stored AT this node
    uint32   level             octree depth (0 = root)
    uint32   subtreeCount      points in this node's whole subtree (diagnostics)
    int32    child[8]          node-table indices, -1 where absent
  points (pointCount x 8 B)
    uint16   qx, qy, qz        position inside the node cube, 0..65535
    uint8    qs               size proxy, 0..254 over [smin, smax]
    uint8    qv               value, 0..254 over [vmin, vmax]; 255 = not measured

qv = 255 is the explicit "no measured value" sentinel. The viewer must colour
those points neutrally rather than inventing a value, same contract as v2's NaN.

Input is the existing .bin files (v1 headerless Float32[x,y,z,r] or v2
Float32[x,y,z,size,value]), so no catalogue refetch is needed to repack.

Usage:  python3 pack_octree.py
"""

import os
import struct
import sys

import numpy as np

CAPACITY = 8000      # points kept at a single node before it subdivides
MAX_DEPTH = 12
NODE_BYTES = 64
HEADER_BYTES = 32

JOBS = [
    # (source .bin, destination .bin)
    ("stars.bin", "stars_o.bin"),
    ("quasars.bin", "quasars_o.bin"),
    ("galaxies.bin", "galaxies_o.bin"),
    ("exo.bin", "exo_o.bin"),
]


def read_v1_v2(path):
    """Decode a v1 or v2 point file into (pos[n,3], size[n], value[n])."""
    raw = open(path, "rb").read()
    if raw[:4] == b"CSMS":
        version, n, stride = struct.unpack("<3I", raw[4:16])
        if version != 2:
            print(f"  !! {path}: header says version {version}, reading as stride {stride}")
        f = np.frombuffer(raw, dtype="<f4", offset=16, count=n * stride).reshape(n, stride)
        value = f[:, 4].astype(np.float64) if stride >= 5 else np.full(n, np.nan)
    else:
        # v1: bare Float32 [x, y, z, r], no header, no measured scalar
        f = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)
        n, stride = f.shape[0], 4
        value = np.full(n, np.nan)
        print(f"  {path}: v1 file (no header) — no measured scalar present, value = NaN")
    pos = f[:, 0:3].astype(np.float64)
    size = f[:, 3].astype(np.float64)
    return pos, size, value


def qrange(a, pct=None):
    """Dequantization range over the finite values.

    `pct` clips to a percentile pair — used for the size/brightness proxy, where a
    handful of extreme radii would otherwise squash every real value into two of
    the 255 levels. The measured `value` column uses the FULL min/max instead: a
    1st-percentile clip would saturate the rare hot-blue tail of bp_rp, i.e. throw
    away exactly the stars whose colour is most visible.
    """
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    if pct is None:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = np.percentile(finite, list(pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


class Node:
    __slots__ = ("bmin", "edge", "level", "idx", "children", "subtree")

    def __init__(self, bmin, edge, level):
        self.bmin = bmin          # np.float64[3]
        self.edge = float(edge)
        self.level = int(level)
        self.idx = np.empty(0, dtype=np.int64)   # points kept AT this node
        self.children = [None] * 8
        self.subtree = 0


def build(pos, rng):
    """Build the octree. Returns (root, node_list) in depth-first order."""
    n = pos.shape[0]
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    centre = (lo + hi) / 2.0
    edge = float((hi - lo).max())
    if edge <= 0:
        edge = 1.0
    edge *= 1.0001                       # keep max-corner points strictly inside
    root = Node(centre - edge / 2.0, edge, 0)

    nodes = []

    def recurse(node, idx):
        node.subtree = int(idx.size)
        nodes.append(node)
        if idx.size <= CAPACITY or node.level >= MAX_DEPTH:
            node.idx = idx
            return
        # random subsample stays at this node; it IS the coarse LOD for the region
        perm = rng.permutation(idx.size)
        node.idx = idx[perm[:CAPACITY]]
        rest = idx[perm[CAPACITY:]]

        half = node.edge / 2.0
        p = pos[rest] - node.bmin
        octant = ((p[:, 0] >= half).astype(np.int64)
                  | ((p[:, 1] >= half).astype(np.int64) << 1)
                  | ((p[:, 2] >= half).astype(np.int64) << 2))
        for o in range(8):
            sel = rest[octant == o]
            if sel.size == 0:
                continue
            off = np.array([o & 1, (o >> 1) & 1, (o >> 2) & 1], dtype=np.float64) * half
            child = Node(node.bmin + off, half, node.level + 1)
            node.children[o] = child
            recurse(child, sel)

    recurse(root, np.arange(n, dtype=np.int64))
    return root, nodes


def pack(src, dst, seed=12345):
    if not os.path.exists(src):
        print(f"skip {src} (absent)")
        return
    pos, size, value = read_v1_v2(src)
    n = pos.shape[0]
    finite = np.isfinite(pos).all(axis=1)
    if not finite.all():
        print(f"  {src}: dropping {int((~finite).sum()):,} points with non-finite positions")
        pos, size, value = pos[finite], size[finite], value[finite]
        n = pos.shape[0]

    rng = np.random.default_rng(seed)
    root, nodes = build(pos, rng)
    order = {id(nd): i for i, nd in enumerate(nodes)}

    smin, smax = qrange(size, pct=(1.0, 99.0))
    vmin, vmax = qrange(value)

    # ── points, grouped node by node in node-table order ──────────────────
    total = sum(int(nd.idx.size) for nd in nodes)
    assert total == n, f"octree lost points: {total} != {n}"
    pts = np.empty((total, 4), dtype=np.uint16)   # qx,qy,qz packed here; qs/qv appended below
    small = np.empty((total, 2), dtype=np.uint8)
    offsets = []
    cursor = 0
    for nd in nodes:
        idx = nd.idx
        c = int(idx.size)
        offsets.append(cursor)
        if c:
            p = (pos[idx] - nd.bmin) / nd.edge
            np.clip(p, 0.0, 1.0, out=p)
            pts[cursor:cursor + c, 0:3] = np.rint(p * 65535.0).astype(np.uint16)
            s = (size[idx] - smin) / (smax - smin)
            s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0)
            small[cursor:cursor + c, 0] = np.rint(s * 254.0).astype(np.uint8)
            v = value[idx]
            qv = np.full(c, 255, dtype=np.uint8)
            m = np.isfinite(v)
            if m.any():
                t = np.clip((v[m] - vmin) / (vmax - vmin), 0.0, 1.0)
                qv[m] = np.rint(t * 254.0).astype(np.uint8)
            small[cursor:cursor + c, 1] = qv
        cursor += c

    # ── serialize ─────────────────────────────────────────────────────────
    out = bytearray()
    out += b"CSM3"
    out += struct.pack("<3I", 3, len(nodes), total)
    out += struct.pack("<4f", vmin, vmax, smin, smax)
    assert len(out) == HEADER_BYTES, len(out)

    for nd, off in zip(nodes, offsets):
        rec = struct.pack("<4f", nd.bmin[0], nd.bmin[1], nd.bmin[2], nd.edge)
        rec += struct.pack("<4I", off, int(nd.idx.size), nd.level, nd.subtree)
        rec += struct.pack("<8i", *[order[id(c)] if c is not None else -1 for c in nd.children])
        assert len(rec) == NODE_BYTES, len(rec)
        out += rec

    body = np.empty((total, 8), dtype=np.uint8)
    body[:, 0:6] = pts[:, 0:3].copy().view(np.uint8).reshape(total, 6)
    body[:, 6:8] = small
    out += body.tobytes()

    with open(dst, "wb") as f:
        f.write(out)

    depths = [nd.level for nd in nodes]
    leaves = sum(1 for nd in nodes if all(c is None for c in nd.children))
    measured = int((small[:, 1] != 255).sum())
    src_kb = os.path.getsize(src) // 1024
    dst_kb = os.path.getsize(dst) // 1024
    print(f"{dst}: {total:,} points, {len(nodes):,} nodes "
          f"(depth 0..{max(depths)}, {leaves:,} leaves), {dst_kb:,} KB "
          f"vs {src_kb:,} KB  ({dst_kb / max(src_kb, 1):.2f}x)")
    print(f"    value range [{vmin:.4g}, {vmax:.4g}] — {measured:,}/{total:,} measured, "
          f"{total - measured:,} flagged unmeasured (255)")
    print(f"    root points {int(root.idx.size):,} = coarsest LOD; "
          f"size range [{smin:.4g}, {smax:.4g}]")


if __name__ == "__main__":
    for src, dst in JOBS:
        pack(src, dst)
    if "--keep" not in sys.argv:
        print("\nWrote *_o.bin. The v2/v1 files are left in place for comparison.")
