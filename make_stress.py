"""Synthetic stress files for the renderer benchmark. NOT data. NOT shipped.

The point budget question is "what happens at 4 million points", and the real
catalogues on disk only add up to 849,918. This inflates the star layer by
jittering copies of real Gaia positions so the renderer sees a realistically
clustered 4M-point cloud — the geometry is fake, and nothing here is ever loaded
by the shipped viewer. The output names start with `stress_` for that reason.

Writes both formats so the comparison is like for like:
  stress_v2.bin  flat Float32 [x,y,z,size,value]   (what the old renderer eats)
  stress_v3.bin  quantized octree                  (what the new renderer eats)

Usage:  python3 make_stress.py [multiplier]     # default 8 → ~4.0M points
"""

import os
import struct
import sys

import numpy as np

import pack_octree as PO

SRC = "stars.bin"


def main():
    mult = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pos, size, value = PO.read_v1_v2(SRC)
    n = pos.shape[0]
    rng = np.random.default_rng(7)

    # jitter scale = 1% of the cloud's extent, so copies land near their parent
    # and the clustering statistics stay roughly realistic for a fill-rate test
    extent = float((pos.max(axis=0) - pos.min(axis=0)).max())
    jit = extent * 0.01

    parts_p, parts_s, parts_v = [pos], [size], [value]
    for _ in range(mult - 1):
        parts_p.append(pos + rng.normal(0.0, jit, size=pos.shape))
        parts_s.append(size)
        parts_v.append(value)
    P = np.concatenate(parts_p)
    S = np.concatenate(parts_s)
    V = np.concatenate(parts_v)
    total = P.shape[0]
    print(f"synthetic cloud: {total:,} points ({mult}x {n:,} real Gaia positions, "
          f"jittered by {jit:.1f} pc)")

    # ── v2 flat file ──────────────────────────────────────────────────────
    flat = np.empty((total, 5), dtype="<f4")
    flat[:, 0:3] = P
    flat[:, 3] = S
    flat[:, 4] = V
    with open("stress_v2.bin", "wb") as f:
        f.write(b"CSMS")
        f.write(struct.pack("<3I", 2, total, 5))
        f.write(flat.tobytes())
    print(f"stress_v2.bin: {os.path.getsize('stress_v2.bin') / 1048576:.1f} MB")

    # ── v3 octree file, via the real packer ───────────────────────────────
    tmp = "stress_src.bin"
    with open(tmp, "wb") as f:
        f.write(b"CSMS")
        f.write(struct.pack("<3I", 2, total, 5))
        f.write(flat.tobytes())
    PO.pack(tmp, "stress_v3.bin")
    os.remove(tmp)


if __name__ == "__main__":
    main()
