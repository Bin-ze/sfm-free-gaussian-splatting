# -*- coding: utf-8 -*-
# Author: Bin-ze
# Email: binze.zero@gmail.com
# Date: 2024/5/29 15:24
# File Name: visual_gs_camear.py

"""
Description of the script or module goes here.
# 可视化重定位相机在地面平行坐标系下的位置
"""

# Your code starts here
import open3d as o3d
import numpy as np
from visual_utils import visual_open3d
import open3d.visualization.gui as gui

def open3d_vis_sys(name='vis'):
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(f"{name}", 1320, 989)
    vis.show_settings = True
    return  app, vis

app, vis = open3d_vis_sys()

# 修改这里
c2ws = [[[-0.9982705155170057, -0.00771644448733319, 0.058278935593260424, -5.244967355502076], [0.0027843330554434545, 0.9840294177476128, 0.17798413551979975, -1.9885794166603596], [-0.05872159176015117, 0.1778385826860072, -0.982306068987352, 3.2892247761429974], [0.0, 0.0, 0.0, 1.0]], [[0.9963544556938227, -0.005819754964022691, -0.08508905178564861, -4.646710750138108], [0.017341116031716994, 0.9906504902141525, 0.13531337040335634, -1.943298808411621], [0.08350526731250917, -0.13629734861633688, 0.9871397763115517, -12.057658338784679], [0.0, 0.0, 0.0, 1.0]], [[0.8600644798093686, 0.15394134468937654, -0.4863959258621319, -5.814323263519019], [0.012005664099329497, 0.9470116093260031, 0.3209595950795016, -2.0165595627684856], [0.5100299594805985, -0.28189232512991214, 0.8126431762390622, -14.510267380470978], [0.0, 0.0, 0.0, 1.0]], [[-0.9584143265942876, 0.12445413637039551, -0.2567936725251079, -4.497103183514039], [0.02824423981909833, 0.9368292841886976, 0.3486306656704542, -1.9782680380050466], [0.28396302428825493, 0.32687378724599414, -0.9013885413468322, -12.21723546750505], [0.0, 0.0, 0.0, 1.0]], [[-0.9298035984882237, 0.1225272980060273, -0.34704779672475544, -5.986543492776594], [0.02075164186022369, 0.9589023556836728, 0.2829590561666085, -2.021689566864306], [0.3674579113521937, 0.2558884072678019, -0.8941356613140267, -10.727232468364075], [0.0, 0.0, 0.0, 1.0]], [[0.9974930078528222, -0.007112159865940311, -0.07022531645592335, -4.198154329827391], [0.02551046399507896, 0.9639536642204546, 0.2647979623787838, -1.9965092257098533], [0.0658083914062132, -0.26593600332516715, 0.9617314431977487, -23.020130872708446], [0.0, 0.0, 0.0, 1.0]]]
c2ws = np.array(c2ws)
gs_path = "/Users/binze/Desktop/5f_sub_room_project_1/gs_trans_0528.ply"

geoms = []
# 0. add origin world frame
origin_w_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
geoms.append(origin_w_frame)

# 1.1 add global point cloud
ply = o3d.io.read_point_cloud(gs_path)
ply.uniform_down_sample(1000)
vis.add_geometry(f"pcd_scene", ply)

# 1.2 visual camera
intrinsics = np.array([
    [2000, 0, 1000],
    [0, 2000, 1000],
    [0, 0, 1]
])

for i, c2w in enumerate(c2ws):
    intr = intrinsics
    extr = c2w
    cam_geoms = visual_open3d.create_camera_frustum(intr, extr @ visual_open3d.OPEN3D_ARKit_transform, color=[0, 1, 0], scale=0.1, frustum_scale=1000)
    vis.add_geometry(f"cam_{i}", cam_geoms[0])
    vis.add_geometry(f"cam_coord_{i}", cam_geoms[1])

# visualize
vis.reset_camera_to_default()
app.add_window(vis)
app.run()
