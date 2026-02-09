# milky_way_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy import units as u

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Map of the Universe",
    layout="wide",
)

st.markdown(
    """
    <style>
    body { background-color: black; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌 Map of the Universe")

# ---------------- SCALE SELECTOR ----------------
scale = st.selectbox(
    "Choose scale",
    [
        "Solar Neighborhood",
        "Milky Way (Spiral View)",
    ]
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("gaia_sample.csv")

required_cols = ["parallax", "l", "b"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"Missing column: {c}")
        st.stop()

df = df.dropna(subset=required_cols)
df = df[df["parallax"] > 0]

# ---------------- DISTANCE + COORDINATES ----------------
df["distance_pc"] = 1000 / df["parallax"]

coords = SkyCoord(
    l=df["l"].values * u.deg,
    b=df["b"].values * u.deg,
    distance=df["distance_pc"].values * u.pc,
    frame="galactic",
)

df["x"] = coords.cartesian.x.value
df["y"] = coords.cartesian.y.value
df["z"] = coords.cartesian.z.value
df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)

# ---------------- SPIRAL ARM SHAPING ----------------
theta = np.arctan2(df["y"], df["x"])
spiral_offset = np.sin(4 * theta + df["r"] / 1500)
df["spiral_weight"] = np.exp(-df["z"]**2 / 200**2) * (1 + 0.6 * spiral_offset)

# ---------------- MAIN GALAXY VIEW ----------------
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=dict(
            size=1.6,
            color=df["r"],
            colorscale="Inferno",
            opacity=0.65,
        ),
        name="Stars",
    )
)

# ---------------- FOG / DENSITY GLOW ----------------
fog = df.sample(min(4000, len(df)))

fig.add_trace(
    go.Scatter3d(
        x=fog["x"],
        y=fog["y"],
        z=fog["z"],
        mode="markers",
        marker=dict(
            size=4,
            color="white",
            opacity=0.03,
        ),
        showlegend=False,
    )
)

fig.update_layout(
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        bgcolor="black",
        camera=dict(
            eye=dict(x=1.4, y=1.4, z=0.6)
        ),
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MINI MAP (TOP-DOWN) ----------------
st.subheader("🗺️ Galaxy Overview (Top-Down)")

mini = go.Figure()

mini.add_trace(
    go.Scattergl(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=1,
            color=df["r"],
            colorscale="Inferno",
            opacity=0.5,
        ),
    )
)

mini.update_layout(
    xaxis_visible=False,
    yaxis_visible=False,
    paper_bgcolor="black",
    plot_bgcolor="black",
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(mini, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<center style='color:gray'>Gaia data • Cinematic galaxy rendering</center>",
    unsafe_allow_html=True,
)
