"""CSV -> packed binary for the viewer.

v2 format. v1 packed Float32 [x,y,z,r] and discarded every physical quantity the
pipeline had already downloaded — which is why star colour in the viewer came from
a hash of the array index rather than from Gaia. v2 keeps one real scalar per point:

    header : magic "CSMS" | uint32 version | uint32 count | uint32 stride_floats
    record : float32 x, y, z, size, value

`value` is that layer's measured quantity, so the shader can colour by physics:
    stars     bp_rp    (Gaia colour index, ~-0.5 blue … ~5 red)
    galaxies  redshift
    quasars   redshift
    exo       none yet -> NaN

NaN means "not measured"; the shader must fall back to a neutral colour rather than
invent one. v1 also took size from the LAST column, which would now silently pick up
bp_rp — v2 addresses columns by name instead.
"""
import csv, struct, os

MAGIC = b"CSMS"
VERSION = 2
STRIDE = 5  # x, y, z, size, value

JOBS = [
    # (csv, bin, column holding the real scalar)
    ("gaia_points.csv",       "stars.bin",    "bp_rp"),
    ("quasars_points.csv",    "quasars.bin",  "redshift"),
    ("galaxies_points.csv",   "galaxies.bin", "redshift"),
    ("exoplanets_points.csv", "exo.bin",      None),
]


def col_index(header, name):
    """Index of `name` in header, or None. Tolerates case/whitespace drift."""
    if not name:
        return None
    norm = [h.strip().lower() for h in header]
    return norm.index(name) if name in norm else None


def convert(src, dst, value_col):
    if not os.path.exists(src):
        print(f"skip {src} (absent)")
        return

    with open(src) as f:
        r = csv.reader(f)
        header = next(r, None) or []
        vi = col_index(header, value_col)
        si = col_index(header, "r")
        if value_col and vi is None:
            print(f"  !! {src}: no '{value_col}' column — value stays NaN "
                  f"(neutral colour, never invented)")

        rows = bytearray()
        n = missing = 0
        for row in r:
            if len(row) < 3:
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
            except ValueError:
                continue
            if x != x or y != y or z != z:
                continue

            size = 1.0
            if si is not None and si < len(row):
                try:
                    size = float(row[si])
                except ValueError:
                    size = 1.0

            value = float("nan")
            if vi is not None and vi < len(row):
                try:
                    value = float(row[vi])
                except ValueError:
                    pass
            if value != value:
                missing += 1

            rows += struct.pack("<5f", x, y, z, size, value)
            n += 1

    with open(dst, "wb") as o:
        o.write(MAGIC)
        o.write(struct.pack("<3I", VERSION, n, STRIDE))
        o.write(rows)

    kb = os.path.getsize(dst) // 1024
    note = f", {missing:,} without a measured value" if missing else ""
    print(f"{dst}: {n:,} points, {kb:,} KB{note}")


if __name__ == "__main__":
    for src, dst, col in JOBS:
        convert(src, dst, col)
