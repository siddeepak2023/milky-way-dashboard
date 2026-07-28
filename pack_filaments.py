"""filaments_points.csv -> filaments.bin (format "CSMF" v2).

The cosmic web is a line list, not a point cloud, so it does not use the CSM3
octree format. Each row of the CSV is one segment of a filament spine, already
paired by prepare_universe_data.load_filaments().

    header  : magic "CSMF" | uint32 version, segCount | float32 bmin[3], edge
    segment : uint16 x1,y1,z1,x2,y2,z2 | uint8 t1,t2   (14 B/segment)

Positions are quantized to a single global cube rather than per-node as CSM3 does.
The SDSS filament volume spans a few hundred Mpc, so one uint16 grid over the whole
bounding cube resolves ~0.016 Mpc — far finer than the ~1 Mpc thickness of a real
filament, and finer than anything the viewer can draw at these zoom levels.

v2 adds t1/t2: each endpoint's normalised arc position along the filament it belongs
to, 0 at one end and 1 at the other, quantized to a byte. That is what lets the
shader run a travelling pulse down a spine without knowing the topology — it reads a
per-vertex number instead of walking the polyline every frame.

Filament grouping is recovered from the CSV rather than stored: within a filament the
segments are consecutive and each one's start point equals the previous one's end
point, so a break in that chain is a filament boundary. Exact equality is the right
test here because both values are written from the same float, not recomputed.

What is NOT packed: the Bisous visit-map strength (`fden`) and filament length
(`Len`). Both are real columns in the catalogue and both would be better than a flat
colour, but the CSV shipped here predates them and VizieR was unreachable when this
was written. The loader in prepare_universe_data.py now selects them; once it has
been run end to end, this packer should carry `fden` as a per-segment byte and the
shader should modulate brightness by it.
"""
import csv, struct, sys, os, math

MAGIC = b"CSMF"
VERSION = 2
SRC = "filaments_points.csv"
DST = "filaments.bin"
COLS = ("x1", "y1", "z1", "x2", "y2", "z2")


def read_segments(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip().lower() for h in next(r)]
        try:
            idx = [header.index(c) for c in COLS]
        except ValueError:
            sys.exit(f"{path} must have columns {COLS}, found {header}")
        out = []
        for row in r:
            if not row:
                continue
            try:
                v = [float(row[i]) for i in idx]
            except (ValueError, IndexError):
                continue                      # skip malformed rather than write a wrong point
            if any(x != x for x in v):
                continue                      # NaN
            out.append(v)
        return out


def arc_params(segs):
    """Per-endpoint normalised arc position, walking each filament chain."""
    n = len(segs)
    t1 = [0.0] * n
    t2 = [0.0] * n
    i = 0
    runs = 0
    while i < n:
        # extent of this filament: while each segment continues the previous one
        j = i
        while j + 1 < n and segs[j + 1][0:3] == segs[j][3:6]:
            j += 1
        # cumulative length over segments i..j
        cum = [0.0]
        for k in range(i, j + 1):
            a, b = segs[k][0:3], segs[k][3:6]
            cum.append(cum[-1] + math.dist(a, b))
        total = cum[-1] or 1.0
        for k in range(i, j + 1):
            t1[k] = cum[k - i] / total
            t2[k] = cum[k - i + 1] / total
        runs += 1
        i = j + 1
    return t1, t2, runs


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found — run prepare_universe_data.py first")

    segs = read_segments(SRC)
    if not segs:
        sys.exit(f"{SRC} contained no usable segments")

    lo = [min(min(s[k], s[k + 3]) for s in segs) for k in range(3)]
    hi = [max(max(s[k], s[k + 3]) for s in segs) for k in range(3)]
    # One cube for all three axes, so the quantization step is isotropic and a
    # segment's direction is not distorted by a per-axis scale.
    edge = max(hi[k] - lo[k] for k in range(3)) or 1.0
    q = 65535.0 / edge

    t1, t2, runs = arc_params(segs)

    with open(DST, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", VERSION, len(segs)))
        out.write(struct.pack("<ffff", lo[0], lo[1], lo[2], edge))
        packer = struct.Struct("<6H2B").pack
        for s, a, b in zip(segs, t1, t2):
            out.write(packer(
                int((s[0] - lo[0]) * q), int((s[1] - lo[1]) * q), int((s[2] - lo[2]) * q),
                int((s[3] - lo[0]) * q), int((s[4] - lo[1]) * q), int((s[5] - lo[2]) * q),
                int(a * 255), int(b * 255),
            ))

    mb = os.path.getsize(DST) / 1048576
    print(f"{DST}: {len(segs):,} segments in {runs:,} filament chains, {mb:.2f} MB")
    print(f"  cube edge {edge:.4g} pc  ->  step {edge/65535:.4g} pc "
          f"({edge/65535/1e6:.4g} Mpc)")


if __name__ == "__main__":
    main()
