import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Universe Explorer", layout="wide", page_icon="🌌")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #020408 !important;
    color: #c8d8f0;
    font-family: 'Space Grotesk', sans-serif;
}
.title-block {
    text-align: center;
    padding: 20px 0 4px 0;
}
.title-block h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #ffffff;
    margin: 0;
}
.title-block p {
    color: #6888aa;
    font-size: 0.88rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
}
.stat-row {
    display: flex;
    justify-content: center;
    gap: 32px;
    padding: 12px 0 20px 0;
}
.stat-item {
    text-align: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 10px 24px;
}
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #88c0f8;
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #445566;
    margin-top: 2px;
}
.stSidebar { background-color: #080e18 !important; }
.stSidebar .stMarkdown { color: #8899aa; }
section[data-testid="stSidebar"] { border-right: 1px solid #0d1a2a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA  (all normalised to parsecs)
# ─────────────────────────────────────────
@st.cache_data
def load_all():
    stars = pd.read_csv("gaia_points.csv")
    exo   = pd.read_csv("exoplanets_points.csv")
    gal   = pd.read_csv("galaxies_points.csv")

    # Galaxies are stored in pc × 1e6 (i.e. Mpc * 1e6 pc/Mpc).
    # Convert to Mpc for a sensible common coordinate space.
    # Stars / exoplanets are in pc → convert to Mpc too.
    PC_TO_MPC = 1 / 1_000_000

    stars["xm"] = stars["x"] * PC_TO_MPC
    stars["ym"] = stars["y"] * PC_TO_MPC
    stars["zm"] = stars["z"] * PC_TO_MPC
    stars["rm"] = stars["r"] * PC_TO_MPC

    exo["xm"] = exo["x"] * PC_TO_MPC
    exo["ym"] = exo["y"] * PC_TO_MPC
    exo["zm"] = exo["z"] * PC_TO_MPC

    # Galaxies: stored in units of pc (= Mpc * 1e6), divide by 1e6 → Mpc
    gal["xm"] = gal["x"] / 1_000_000
    gal["ym"] = gal["y"] / 1_000_000
    gal["zm"] = gal["z"] / 1_000_000
    gal["rm"] = gal["r"] / 1_000_000

    return stars, exo, gal

stars, exo, gal = load_all()

# ─────────────────────────────────────────
# FAST FILAMENTS via cKDTree  (k=3 neighbours)
# ─────────────────────────────────────────
@st.cache_data
def build_filaments(n_gal=2000, k=3):
    sub = gal.nsmallest(n_gal, "rm")[["xm","ym","zm"]].values
    fx, fy, fz = [], [], []
    seen = set()

    if HAS_SCIPY:
        tree = cKDTree(sub)
        _, idxs = tree.query(sub, k=k+1)
        for i in range(len(sub)):
            for j in idxs[i, 1:]:
                key = (min(i,j), max(i,j))
                if key in seen: continue
                seen.add(key)
                fx += [sub[i,0], sub[j,0], None]
                fy += [sub[i,1], sub[j,1], None]
                fz += [sub[i,2], sub[j,2], None]
    else:
        # Numpy fallback: find single nearest neighbour per point
        for i in range(len(sub)):
            diff = sub - sub[i]
            dists = np.sum(diff**2, axis=1)
            dists[i] = np.inf
            j = int(np.argmin(dists))
            key = (min(i,j), max(i,j))
            if key in seen: continue
            seen.add(key)
            fx += [sub[i,0], sub[j,0], None]
            fy += [sub[i,1], sub[j,1], None]
            fz += [sub[i,2], sub[j,2], None]

    return fx, fy, fz

# ─────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────
st.sidebar.markdown("### 🌌 Layer Controls")

show_stars    = st.sidebar.checkbox("Gaia Stars",           True)
show_exo      = st.sidebar.checkbox("Exoplanets",           True)
show_galaxies = st.sidebar.checkbox("Galaxies (2MRS)",      True)
show_filaments= st.sidebar.checkbox("Cosmic Web Filaments", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Style")
star_size   = st.sidebar.slider("Star size",    0.5, 5.0, 1.5, 0.5)
gal_size    = st.sidebar.slider("Galaxy size",  0.5, 8.0, 3.0, 0.5)
exo_size    = st.sidebar.slider("Exoplanet size", 0.5, 6.0, 2.5, 0.5)
fil_opacity = st.sidebar.slider("Filament opacity", 0.1, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Camera Presets")
cam_preset = st.sidebar.radio("View", [
    "🌍 Local neighbourhood",
    "🌌 Galactic scale",
    "🕸️ Cosmic web",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Stats")
st.sidebar.markdown(f"""
<div style='font-family:monospace;font-size:12px;color:#446688;line-height:1.8'>
Stars &nbsp;&nbsp;&nbsp;&nbsp;: {len(stars):,}<br>
Exoplanets: {len(exo):,}<br>
Galaxies &nbsp;: {len(gal):,}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class='title-block'>
  <h1>🌌 UNIVERSE EXPLORER</h1>
  <p>Gaia DR3 Stars &nbsp;·&nbsp; 2MRS Galaxies &nbsp;·&nbsp; NASA Exoplanets &nbsp;·&nbsp; Cosmic Web</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='stat-row'>
  <div class='stat-item'><div class='stat-num'>{len(stars):,}</div><div class='stat-label'>Gaia Stars</div></div>
  <div class='stat-item'><div class='stat-num'>{len(gal):,}</div><div class='stat-label'>Galaxies</div></div>
  <div class='stat-item'><div class='stat-num'>{len(exo):,}</div><div class='stat-label'>Exoplanets</div></div>
  <div class='stat-item'><div class='stat-num'>~740 Mpc</div><div class='stat-label'>Max Depth</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# COLOUR FUNCTIONS
# ─────────────────────────────────────────
def star_colors(rm):
    n = np.clip(rm / np.percentile(rm, 95), 0, 1)
    r = (170 + 85*n).astype(int)
    g = (190 + 50*n).astype(int)
    b = np.full_like(r, 255)
    return [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]

def galaxy_colors(rm):
    n = np.clip(rm / np.percentile(rm, 95), 0, 1)
    r = np.full(len(n), 255, dtype=int)
    g = (70 + 90*(1-n)).astype(int)
    b = (190 + 65*n).astype(int)
    return [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]

# ─────────────────────────────────────────
# CAMERA PRESET LOGIC
# ─────────────────────────────────────────
axis_limit = float(np.percentile(gal["rm"], 99) * 1.1)

if cam_preset == "🌍 Local neighbourhood":
    # Zoom into star / exoplanet scale (~0.05 Mpc)
    eye_scale = 0.0002
    cam_range = 0.1
elif cam_preset == "🌌 Galactic scale":
    eye_scale = 0.05
    cam_range = axis_limit * 0.15
else:  # Cosmic web
    eye_scale = 0.4
    cam_range = axis_limit

theta = np.deg2rad(40)
camera = dict(
    eye=dict(
        x=cam_range * np.cos(theta),
        y=cam_range * np.sin(theta),
        z=cam_range * 0.6,
    ),
    projection=dict(type="perspective")
)

# ─────────────────────────────────────────
# BUILD FIGURE
# ─────────────────────────────────────────
fig = go.Figure()

if show_filaments:
    with st.spinner("Building cosmic web filaments…"):
        fx, fy, fz = build_filaments(n_gal=2000, k=3)
    fig.add_trace(go.Scatter3d(
        x=fx, y=fy, z=fz,
        mode="lines",
        line=dict(width=1, color=f"rgba(100,200,255,{fil_opacity})"),
        hoverinfo="none",
        name="Cosmic web"
    ))

if show_stars:
    sc = star_colors(stars["rm"].values)
    fig.add_trace(go.Scatter3d(
        x=stars["xm"], y=stars["ym"], z=stars["zm"],
        mode="markers",
        marker=dict(size=star_size, color=sc, opacity=0.92),
        hovertemplate="<b>Star</b><br>Distance: %{customdata:.1f} pc<extra></extra>",
        customdata=(stars["rm"] * 1_000_000).values,
        name="Gaia Stars"
    ))

if show_galaxies:
    gc = galaxy_colors(gal["rm"].values)
    fig.add_trace(go.Scatter3d(
        x=gal["xm"], y=gal["ym"], z=gal["zm"],
        mode="markers",
        marker=dict(size=gal_size, color=gc, opacity=0.88),
        hovertemplate="<b>Galaxy</b><br>Distance: %{customdata:.0f} Mpc<extra></extra>",
        customdata=gal["rm"].values,
        name="2MRS Galaxies"
    ))

if show_exo:
    exo_s = exo.sample(min(5000, len(exo)), random_state=42)
    fig.add_trace(go.Scatter3d(
        x=exo_s["xm"], y=exo_s["ym"], z=exo_s["zm"],
        mode="markers",
        marker=dict(size=exo_size, color="rgba(0,255,220,0.85)", opacity=0.85),
        hoverinfo="none",
        name="Exoplanets"
    ))

fig.update_layout(
    scene=dict(
        bgcolor="#010509",
        xaxis=dict(visible=False, range=[-axis_limit, axis_limit]),
        yaxis=dict(visible=False, range=[-axis_limit, axis_limit]),
        zaxis=dict(visible=False, range=[-axis_limit, axis_limit]),
    ),
    paper_bgcolor="#020408",
    margin=dict(l=0, r=0, t=0, b=0),
    height=780,
    dragmode="orbit",
    scene_camera=camera,
    showlegend=True,
    legend=dict(
        bgcolor="rgba(2,8,16,0.85)",
        bordercolor="#0d2040",
        borderwidth=1,
        font=dict(color="#8899bb", family="Space Mono", size=11),
        x=0.01, y=0.98
    ),
    uirevision="stable",
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["toImage"],
})

# ─────────────────────────────────────────
# SCALE CONTEXT
# ─────────────────────────────────────────
st.markdown("""
<div style='display:flex;gap:16px;padding:8px 0 0 0;flex-wrap:wrap;'>
""", unsafe_allow_html=True)

cols = st.columns(4)
with cols[0]:
    st.markdown("**🔵 Gaia Stars**")
    st.caption(f"Milky Way neighbourhood · {len(stars):,} sources · up to ~10,600 pc · Blue = farther")
with cols[1]:
    st.markdown("**🟣 Galaxies**")
    st.caption(f"2MRS survey · {len(gal):,} galaxies · up to ~740 Mpc · Pink-purple gradient by distance")
with cols[2]:
    st.markdown("**🩵 Exoplanets**")
    st.caption(f"NASA Archive · {len(exo):,} systems · Cyan dots within ~6,000 pc of Earth")
with cols[3]:
    st.markdown("**🕸️ Cosmic Web**")
    st.caption("Nearest-neighbour filaments connecting the 2,000 closest galaxies via cKDTree (k=3)")

st.markdown("""
<p style='font-size:11px;color:#334455;margin-top:16px;font-family:monospace;'>
⚠️ All coordinates are in Mpc. Stars & exoplanets occupy a ~0.01 Mpc region at centre — 
use the "Local neighbourhood" camera preset to resolve them from the galaxy field.
</p>
""", unsafe_allow_html=True)