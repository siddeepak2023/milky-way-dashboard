import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Milky Way 3D",
    layout="wide",
)

st.title("🌌 Milky Way Galaxy Visualization")
st.caption("Gaia-inspired perceptual rendering of the Milky Way")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gaia_sample.csv")  # must include x,y,z or ra/dec/parallax
    return df

df = load_data()

# --------------------------------------------------
# Coordinate prep (if needed)
# --------------------------------------------------
# Assume x,y,z already in parsecs
df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

# --------------------------------------------------
# Spiral disk emphasis
# --------------------------------------------------
disk_scale = 300  # thickness of galactic disk (pc)
df["disk_weight"] = np.exp(-(df["z"]**2) / (2 * disk_scale**2))
df["alpha"] = 0.15 + 0.85 * df["disk_weight"]

# Star size scaling
df["size"] = np.clip(6 / (df["r"] + 50), 0.3, 4)

# --------------------------------------------------
# Camera controls
# --------------------------------------------------
st.sidebar.header("Camera")
cam_x = st.sidebar.slider("Camera X", -2.5, 2.5, 1.6)
cam_y = st.sidebar.slider("Camera Y", -2.5, 2.5, 1.6)
cam_z = st.sidebar.slider("Camera Z", -2.5, 2.5, 1.2)

camera_eye = dict(x=cam_x, y=cam_y, z=cam_z)

# --------------------------------------------------
# Main star field
# --------------------------------------------------
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="r",
    color_continuous_scale="Inferno",
    size="size",
    opacity=0.25
)

# --------------------------------------------------
# Glow layer (soft halo)
# --------------------------------------------------
fig.add_scatter3d(
    x=df["x"],
    y=df["y"],
    z=df["z"],
    mode="markers",
    marker=dict(
        size=df["size"] * 3,
        color="white",
        opacity=0.05
    ),
    showlegend=False
)

# --------------------------------------------------
# Density fog (interstellar haze)
# --------------------------------------------------
fog = df.sample(min(3000, len(df)))

fig.add_scatter3d(
    x=fog["x"],
    y=fog["y"],
    z=fog["z"],
    mode="markers",
    marker=dict(
        size=8,
        color="white",
        opacity=0.02
    ),
    showlegend=False
)

# --------------------------------------------------
# Layout polish
# --------------------------------------------------
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="black",
        aspectmode="data",
        camera=dict(eye=camera_eye)
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale=False
)

fig.update_traces(marker=dict(line=dict(width=0)))

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Mini-map overview
# --------------------------------------------------
with st.expander("🛰️ Galaxy Overview (Top-Down)"):
    top_fig = px.scatter(
        df,
        x="x",
        y="y",
        color="r",
        color_continuous_scale="Inferno",
        opacity=0.3
    )

    top_fig.update_layout(
        height=300,
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False
    )

    st.plotly_chart(top_fig, use_container_width=True)
