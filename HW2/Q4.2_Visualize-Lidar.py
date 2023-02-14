import open3d as o3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
from nuscenes.nuscenes import NuScenes
from matplotlib import cm

nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=True)

# Get lidar data
pcd_name = 'data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605548192.pcd.bin'
scan = np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
color = np.zeros([len(points), 3])

# Visualize Lidar data
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
o3d.visualization.draw_geometries([pcd])


# Visualize and colorize the lidar data by height
points_h = points[:, 2]
points_hmax = np.max(points_h)
points_hmin = np.min(points_h)
cmap_h = plt.get_cmap('viridis')
norm_h = mpl.colors.Normalize(points_hmin, points_hmax)
scalarmap_h = cm.ScalarMappable(norm=norm_h, cmap=cmap_h)
color_h = scalarmap_h.to_rgba(points_h)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points[:, :3])
pcd1.colors = o3d.utility.Vector3dVector(color_h[:, :3])
o3d.visualization.draw_geometries([pcd1])


# Visualize and colorize the lidar data by intensity
points_i = points[:, 3]
points_imax = np.max(points_i)
points_imin = np.min(points_i)
cmap_i = plt.get_cmap('viridis')
norm_i = mpl.colors.Normalize(points_imin, points_imax)
scalarmap_i = cm.ScalarMappable(norm=norm_i, cmap=cmap_i)
color_i = scalarmap_i.to_rgba(points_i)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(color_i[:, :3])
o3d.visualization.draw_geometries([pcd2])

# Get the image semantic info
seg_name = 'data/sets/nuscenes/lidarseg/v1.0-mini/0f92b1a57bf84db0b72c22752662ebe6_lidarseg.bin'
seg = np.fromfile(seg_name, dtype=np.uint8)
color_s = np.zeros([len(seg), 3])
color_s[:, 0] = seg/32
color_s[:, 1] = seg/32
color_s[:, 2] = seg/32

# Visualize and colorize the lidar data by segmentation
points_smax = np.max(seg)
points_smin = np.min(seg)
cmap_s = plt.get_cmap('plasma')
norm_s = mpl.colors.Normalize(points_smin, points_smax)

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points[:, :3])
pcd3.colors = o3d.utility.Vector3dVector(color_s[:, :3])
o3d.visualization.draw_geometries([pcd3])