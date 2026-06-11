import csv, struct, os
# Convert each CSV -> packed Float32 [x,y,z,r] per point. r = last col if >3 cols else 1.
jobs = [
    ("gaia_points.csv","stars.bin"),
    ("quasars_points.csv","quasars.bin"),
    ("galaxies_points.csv","galaxies.bin"),
    ("exoplanets_points.csv","exo.bin"),
]
for src,dst in jobs:
    if not os.path.exists(src):
        print("skip",src); continue
    n=0
    with open(src) as f, open(dst,"wb") as o:
        r=csv.reader(f); next(r,None)
        buf=bytearray()
        for row in r:
            if len(row)<3: continue
            try:
                x=float(row[0]); y=float(row[1]); z=float(row[2])
                rad=float(row[-1]) if len(row)>3 else 1.0
            except: continue
            if x!=x or y!=y or z!=z: continue
            buf+=struct.pack("<4f",x,y,z,rad); n+=1
            if len(buf)>1<<20: o.write(buf); buf=bytearray()
        o.write(buf)
    print(f"{dst}: {n} points, {os.path.getsize(dst)//1024} KB")
