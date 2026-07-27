[README (2).md](https://github.com/user-attachments/files/26105595/README.2.md)
# Universe Explorer

A real-time interactive 3D visualisation of the observable universe, built entirely in the browser using Three.js. Every point rendered is a real astronomical object from a real published catalogue — no procedural generation, no fake data.

---

## What you're looking at

| Layer | Source | Count shipped |
|---|---|---|
| Stars | Gaia DR3 (ESA) | 500,000 |
| Galaxies | 2MRS / HyperLEDA | 43,507 |
| Quasars | SDSS DR17 spectroscopic | 300,000 |
| Exoplanets | NASA Exoplanet Archive | 6,411 |

849,918 points total. The pipeline can also fetch Tempel et al. 2014 (Bisous SDSS)
filaments, but **the viewer does not render them yet** — see the plan.

### What the colours mean

- **Stars** are coloured by their **real measured Gaia `BP−RP` colour index**, converted
  to an effective temperature (Ballesteros 2012) and then to a blackbody colour along the
  Planck locus. Cool M-dwarfs come out deep amber, Sun-like stars cream, hot stars blue-white.
  497,163 of the 500,000 have a measured `BP−RP`; the remaining 2,837 render neutral rather
  than being assigned an invented colour.
- **Galaxies, quasars and exoplanets** currently use **one flat colour per layer** — layer
  identity only, encoding nothing per object. Their `redshift` is fetched by the pipeline but
  not yet packed into the binaries; once it is, they get a real epoch ramp.
- **Brightness** follows each object's packed radius/magnitude proxy. **Per-layer exposure**
  is a display setting, not data: 500,000 additively-blended stars need less energy each than
  6,411 exoplanets do.

Earlier versions of this README described a colour gradient mapping "distance and cosmic
epoch". That was not true — colour was derived from a hash of the array index. It is real now
for stars, and honestly flat for the rest.

### On the layout

Each layer is recentred on its own centroid and rescaled so its 98th-percentile radius maps to
a fixed shell (`normalizeLayer`). That makes the layers legible together, but it means
**relative distances between layers are presentational, not physical**. This is a stylised
atlas of shells, not a metric map of the universe.

---

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy astropy requests tqdm
```

### 2. Download the real data
```bash
python prepare_universe_data.py
```
This downloads ~5–15 minutes of real astronomical data from ESA, NASA, SDSS, HyperLEDA, and VizieR. Outputs:
```
gaia_points.csv
galaxies_points.csv
exoplanets_points.csv
quasars_points.csv
filaments_points.csv
```

### 3. Open the viewer
Open `universe_viewer.html` with VS Code Live Server (right-click → Open with Live Server). All CSVs must be in the same folder.

> **Note:** The viewer uses `fetch()` to load CSVs, so it must be served over HTTP — opening the file directly (`file://`) will not work.

---

## Controls

**Mouse** — click and drag to rotate, scroll to zoom  
**Presets** — bottom buttons snap the camera to the right scale for each data layer  
**Panel** — toggle layers on/off, adjust point sizes, brightness  
**FOV breathe** — subtle cinematic zoom pulse, toggle in FX panel

---

## Architecture

```
universe_viewer.html      — self-contained Three.js viewer (~500 lines)
prepare_universe_data.py  — data pipeline, downloads all real catalogues
*.csv                     — pre-processed point clouds (generated, not committed)
```

**Why pure HTML?** No build step, no npm, no bundler. Drop the files in a folder, serve with Live Server, done. Three.js r128 loaded from CDN.

**Performance design:**
- All point clouds: single `THREE.Points` geometry per layer (1 draw call)
- Filaments: single `THREE.LineSegments` (was 1,200 individual Line objects)
- Pixel ratio capped at 1x on laptops
- FOV breathe via low-pass filter — no per-frame matrix choppiness

**Scale:** Everything is normalised to Mpc. Scale factor `S = 1/1,000,000` converts stored parsec values to scene units. Stars are at ~0.00001 Mpc, galaxies at ~10–600 Mpc, quasars at ~400–8,000 Mpc — all coexist in the same coordinate space.

---

## Data sources

- **Gaia DR3** — https://gea.esac.esa.int/tap-server/tap
- **HyperLEDA** — http://leda.univ-lyon1.fr
- **NASA Exoplanet Archive** — https://exoplanetarchive.ipac.caltech.edu/TAP
- **SDSS DR17** — https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch
- **Tempel et al. 2014** — https://vizier.cds.unistra.fr (J/A+A/566/A1)

All data is publicly available and free to use for research and educational purposes.

---

## What's real vs algorithmic

| Element | Status |
|---|---|
| Star positions | ✅ Real — Gaia parallax measurements |
| Galaxy positions | ✅ Real — measured redshifts |
| Exoplanet host positions | ✅ Real — confirmed detections |
| Quasar positions | ✅ Real — SDSS spectroscopic redshifts |
| Cosmic web filaments | ✅ Real if pipeline run, algorithmic fallback otherwise |
| Star colours | ⚠️ Approximated — spectral type probabilities |
| FOV breathing | ❌ Artistic effect |
