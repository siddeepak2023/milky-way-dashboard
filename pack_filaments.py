"""filaments_points.csv -> filaments.bin (format "CSMF" v1).

The cosmic web is a line list, not a point cloud, so it does not use the CSM3
octree format. Each row of the CSV is one segment of a filament spine, already
paired by prepare_universe_data.load_filaments().

    header  : magic "CSMF" | uint32 version, segCount | float32 bmin[3], edge
    segment : uint16 x1,y1,z1,x2,y2,z2   (12 B/segment)

Positions are quantized to a single global cube rather than per-node as CSM3 does.
The SDSS filament volume spans a few hundred Mpc, so one uint16 grid over the whole
bounding cube resolves ~0.01 Mpc — far finer than the ~1 Mpc thickness of a real
filament, and finer than anything the viewer can draw at these zoom levels. A
per-node scheme would buy precision nobody can see.

260,178 segments come to about 3.0 MB, against 6.2 MB for raw float32 pairs.

No per-segment scalar is packed. The catalogue does carry usable ones — filament
length and the Bisous visit-map strength — but this pass renders the network with a
single flat colour, so packing a value the shader ignores would be dead weight.
Colour and brightness here encode filament identity only, nothing per segment.
"""
import csv, struct, sys, os

MAGIC = b"CSMF"
VERSION = 1
SRC = "filaments_points.csv"
DST = "filaments.bin"
COLS = ("x1", "y1", "z1", "x2", "y2", "z2")


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found — run prepare_universe_data.py first")

    with open(SRC, newline="") as f:
        r = csv.reader(f)
        header = [h.strip().lower() for h in next(r)]
        try:
            idx = [header.index(c) for c in COLS]
        except ValueError:
            sys.exit(f"{SRC} must have columns {COLS}, found {header}")

        segs = []
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for row in r:
            if not row:
                continue
            try:
                v = [float(row[i]) for i in idx]
            except (ValueError, IndexError):
                continue          # skip malformed row rather than write a wrong point
            if any(x != x for x in v):
                continue          # NaN
            segs.append(v)
            for a in (0, 3):
                for k in range(3):
                    c = v[a + k]
                    if c < lo[k]:
                        lo[k] = c
                    if c > hi[k]:
                        hi[k] = c

    if not segs:
        sys.exit(f"{SRC} contained no usable segments")

    # One cube for all three axes, so the quantization step is isotropic and a
    # segment's direction is not distorted by a per-axis scale.
    edge = max(hi[k] - lo[k] for k in range(3)) or 1.0
    q = 65535.0 / edge

    with open(DST, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", VERSION, len(segs)))
        out.write(struct.pack("<ffff", lo[0], lo[1], lo[2], edge))
        packer = struct.Struct("<6H").pack
        for v in segs:
            out.write(packer(
                int((v[0] - lo[0]) * q), int((v[1] - lo[1]) * q), int((v[2] - lo[2]) * q),
                int((v[3] - lo[0]) * q), int((v[4] - lo[1]) * q), int((v[5] - lo[2]) * q),
            ))

    mb = os.path.getsize(DST) / 1048576
    print(f"{DST}: {len(segs):,} segments, {mb:.2f} MB")
    print(f"  cube edge {edge:.4g} pc  ->  step {edge/65535:.4g} pc "
          f"({edge/65535/1e6:.4g} Mpc)")


if __name__ == "__main__":
    main()
