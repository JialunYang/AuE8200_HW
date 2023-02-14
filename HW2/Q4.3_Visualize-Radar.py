import open3d as o3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
from nuscenes.nuscenes import NuScenes
from matplotlib import cm

nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=True)

# Visualize the Radar
pcd_name = 'data/sets/nuscenes/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151605526118.pcd'
meta = []
with open(pcd_name, 'rb') as f:
    for line in f:
        line = line.strip().decode('utf-8')
        meta.append(line)
        if line.startswith('DATA'):
            break

    data_binary = f.read()

# Get the header rows and check if they appear as expected.
assert meta[0].startswith('#'), 'First line must be comment'
assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
sizes = meta[3].split(' ')[1:]
types = meta[4].split(' ')[1:]
counts = meta[5].split(' ')[1:]
width = int(meta[6].split(' ')[1])
height = int(meta[7].split(' ')[1])
data = meta[10].split(' ')[1]
feature_count = len(types)
assert width > 0
assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
assert height == 1, 'Error: height != 0 not supported!'
assert data == 'binary'

# Lookup table for how to decode the binaries.
unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                 'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                 'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

# Decode each point.
offset = 0
point_count = width
points = []
for i in range(point_count):
    point = []
    for p in range(feature_count):
        start_p = offset
        end_p = start_p + int(sizes[p])
        assert end_p < len(data_binary)
        point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
        point.append(point_p)
        offset = end_p
    points.append(point)
points = np.asarray(points)

# Visualize the radar data
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
o3d.visualization.draw_geometries([pcd])

# Visualize and colorize the radar data for distance (since the heights are all 0)
dis = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
points_dmax = np.max(dis)
points_dmin = np.min(dis)
cmap_d = plt.get_cmap('plasma')
norm_d = mpl.colors.Normalize(points_dmin, points_dmax)
scalarmap_d = cm.ScalarMappable(norm=norm_d, cmap=cmap_d)
color_d = scalarmap_d.to_rgba(dis)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points[:, :3])
pcd1.colors = o3d.utility.Vector3dVector(color_d[:, :3])
o3d.visualization.draw_geometries([pcd1])

# Visualize and colorize the radar data for velocity
vel = np.sqrt(np.square(points[:, 8]) + np.square(points[:, 9]))
points_vmax = np.max(vel)
points_vmin = np.min(vel)
cmap_v = plt.get_cmap('plasma')
norm_v = mpl.colors.Normalize(points_vmin, points_vmax)
scalarmap_v = cm.ScalarMappable(norm=norm_v, cmap=cmap_v)
color_vel = scalarmap_v.to_rgba(vel)
# Plot the lidar based on the seg
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(color_vel[:, :3])
o3d.visualization.draw_geometries([pcd2])