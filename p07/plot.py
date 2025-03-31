import numpy as np
import plotly.graph_objects as go

# Dimensions
m = 384  # height
n = 512  # width
c = 3    # channels

# Initialize array with signed int type
P = np.zeros((m, n, c), dtype=np.int16)

# Read data from file
with open('peppers.out', 'r') as fid:
    data = [int(x) for x in fid.read().split()]
    idx = 0
    for i in range(m):
        for j in range(n):
            for k in range(c):
                if idx < len(data):
                    P[i, j, k] = data[idx]
                    idx += 1

# Select channel (0-indexed in Python)
channel = 0

# Transformations
z_rot = np.rot90(P[:, :, channel], k=-1)     # 90° clockwise
z_final = np.flipud(z_rot)             # Vertical flip

# Get new dimensions
rot_m, rot_n = z_final.shape
M_final, N_final = np.meshgrid(np.arange(1, rot_n + 1), np.arange(1, rot_m + 1))

# Create the smoothed, flattened 3D surface plot
fig = go.Figure(data=[go.Surface(
    z=z_final,
    x=M_final,
    y=N_final,
    colorscale='Viridis',
    lighting=dict(
        ambient=1,
        diffuse=0,
        specular=0,
        roughness=1,
        fresnel=0
    ),
    showscale=True
)])

fig.update_layout(
    title='Sobel 5x5 Output (Smooth Flattened 3D View, channel 1)',
    scene=dict(
        xaxis_title='Image Width (transformed)',
        yaxis_title='Image Height (transformed)',
        zaxis_title='Pixel Value',
        zaxis=dict(visible=False),  # Hide vertical axis
        camera_eye=dict(x=0.0, y=0.0, z=1.3),  # Top-down view
        aspectmode='manual',
        aspectratio=dict(x=rot_n / rot_m, y=1, z=0.05)  # Flatten the z axis
    ),
    autosize=False,
    width=1280,
    height=768,
    margin=dict(l=65, r=50, b=65, t=90),
)

fig.show()
