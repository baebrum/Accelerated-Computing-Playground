import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def print_usage():
    print("Usage: python plot.py <data_file>")
    print("\n<data_file> should be a text file containing grid data.")
    sys.exit(1)

# Check for the correct number of arguments
if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help']:
    print("Error: Missing or invalid input file.")
    print_usage()

# Load the data from the provided text file
file_path = sys.argv[1]
try:
    data = np.loadtxt(file_path)
except Exception as e:
    print(f"Error loading file '{file_path}': {e}")
    sys.exit(1)

# Create a meshgrid for plotting
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# Create the figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, data, cmap='viridis')

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Rotate the plot by 90 degrees
ax.view_init(elev=25, azim=45)

# Display the plot
plt.show()
