# -*- coding: utf-8 -*-
# Author: Bin-ze
# Email: binze.zero@gmail.com
# Date: 2024/3/5 16:34
# File Name: visual_utils.py

"""
Description of the script or module goes here.
"""

# Your code starts here
import numpy as np
from pathlib import Path
import open3d as o3d
import argparse
import json

class visual_open3d:
    OPEN3D_ARKit_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])  # OPEN3D ARKit uses right-hand coordinate system

    @staticmethod
    def create_bbox(points, color=[0.0, 0.0, 1.0]):
        # 创建线段以表示bbox
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面的四条边
            [4, 5], [5, 6], [6, 7], [7, 4],  # 上面的四条边
            [0, 4], [1, 5], [2, 6], [3, 7]  # 连接底面和上面的四条边
        ]

        color = [color for _ in range(12)]

        # 创建一个线集对象
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
            # colors=o3d.utility.Vector3dVector(color)
        )
        line_set.colors = o3d.utility.Vector3dVector(color)

        return line_set

    @staticmethod
    def create_camera_coordinate_frame(extrinsic, scale=0.1):
        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        cam.transform(extrinsic)

        return cam

    @staticmethod
    def create_camera_frustum(intrinsic, extrinsic, color=[1, 0, 0], scale=0.1, frustum_scale=1000, img_path=None):
        '''
        Create a camera frustum using Open3d LineSet format with given intrinsic and extrinsic matrix.

        Args:
            intrinsic: intrinsic matrix of the camera, in [fx, 0, cx; 0, fy, cy; 0, 0, 1] format
            extrinsic: extrinsic matrix of the camera, in [R, t; 0, 0, 0, 1] format
            color: color of the camera frustum
            scale: scale of the camera frustum
            frustum_scale: scale of the camera plane
            img_path: path to the image of the camera plane

        '''
        cam_geoms = []
        # 相机内参（焦距，主点等）
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # 创建视锥的顶点
        points = [
            [0, 0, 0],  # 相机中心
            [(-cx - frustum_scale) * scale / fx, (-cy - frustum_scale) * scale / fy, -scale],
            [(1 - cx + frustum_scale) * scale / fx, (-cy - frustum_scale) * scale / fy, -scale],
            [(1 - cx + frustum_scale) * scale / fx, (1 - cy + frustum_scale) * scale / fy, -scale],
            [(-cx - frustum_scale) * scale / fx, (1 - cy + frustum_scale) * scale / fy, -scale],
        ]

        # 创建线段以表示视锥边缘
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 连接相机中心和底面的4条边
            [1, 2], [2, 3], [3, 4], [4, 1],  # 底面的4条边
        ]
        # 创建一个线集对象
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # 应用变换矩阵
        line_set.transform(extrinsic)
        # 设置颜色
        colors = [color for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        cam_geoms.append(line_set)

        # 创建三角形网格以表示相机平面
        if img_path:
            # 构建三角形网格
            mesh = o3d.geometry.TriangleMesh()
            # 设置顶点坐标
            mesh.vertices = o3d.utility.Vector3dVector(points)
            # 设置顶点连接关系
            mesh.triangles = o3d.utility.Vector3iVector([[1, 2, 3], [3, 4, 1]])
            # 设置纹理坐标
            mesh.triangle_uvs = o3d.utility.Vector2dVector([
                [0, 1], [1, 1], [1, 0],  # 第一个三角mesh
                [1, 0], [0, 0], [0, 1]  # 第二个三角mesh
            ])
            mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
            # 进行纹理映射
            image = o3d.io.read_image(img_path)
            mesh.textures = [o3d.geometry.Image(image)]
            # 转换三角形网格在空间中的位置
            mesh.transform(extrinsic)
            cam_geoms.append(mesh)

        coord_frame = visual_open3d.create_camera_coordinate_frame(extrinsic)
        cam_geoms.append(coord_frame)
        return cam_geoms

    @staticmethod
    def read_json(transform_file):
        with open(str(transform_file), 'r') as f:
            contents = json.load(f)

        fx = contents["fl_x"]
        fy = contents["fl_y"]
        cx = contents["cx"]
        cy = contents["cy"]

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        H = int(contents["h"])
        W = int(contents["w"])
        # write intrinsic to file
        # write_camera_primesense(str(transform_file.parent), W, H, K)

        # 每次保存相机外参的时候需要删除之前的
        if transform_file.parent.joinpath('trajectory.log').exists():
            transform_file.parent.joinpath('trajectory.log').unlink()
        world_view_transforms = []
        file_names = []
        c2ws = []
        for idx, i in enumerate(contents['frames']):

            # Colmap Coordinate System
            matrix = np.array(i["transform_matrix"])
            c2ws.append(matrix.copy()[None])
            # nerf to colmap
            # matrix[:, 2:3] *= -1
            # format
            matrix_inv = np.linalg.inv(matrix) # w2c
            matrix[:, 3] = matrix_inv[:, 3] # w2c T

            world_view_transform = np.eye(4)
            world_view_transform[:3, :3] = matrix[:3, :3] # c2w R
            world_view_transform[-1, :3] = matrix[:3, -1] # w2c t

            world_view_transforms.append(world_view_transform[None])

            file_names.append(i['file_path'].split('/')[-1])
        # TODO
        world_view_transforms = np.concatenate(world_view_transforms, dtype=np.float32)
        c2ws = np.concatenate(c2ws, dtype=np.float32)


        return H, W, K, world_view_transforms, file_names, c2ws

    @staticmethod
    def read_muti_camera_json(transform_file):
        with open(str(transform_file), 'r') as f:
            contents = json.load(f)
        # write intrinsic to file
        # write_camera_primesense(str(transform_file.parent), W, H, K)

        # 每次保存相机外参的时候需要删除之前的
        if transform_file.parent.joinpath('trajectory.log').exists():
            transform_file.parent.joinpath('trajectory.log').unlink()
        world_view_transforms = []
        c2ws = []
        Ks = []
        file_names = []
        for idx, i in enumerate(contents['frames']):
            # in
            fx = i["fl_x"]
            fy = i["fl_y"]
            cx = i["cx"]
            cy = i["cy"]

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            Ks.append(K[None])
            H = int(i["h"])
            W = int(i["w"])

            # Colmap Coordinate System
            matrix = np.array(i["transform_matrix"])
            c2ws.append(matrix.copy())
            # format
            matrix_inv = np.linalg.inv(matrix) # w2c
            matrix[:, 3] = matrix_inv[:, 3] # w2c T

            world_view_transform = np.eye(4)
            world_view_transform[:3, :3] = matrix[:3, :3] # c2w R
            world_view_transform[-1, :3] = matrix[:3, -1] # w2c t

            world_view_transforms.append(world_view_transform[None])

            file_names.append(i['file_path'].split('/')[-1])
        # TODO
        c2ws = np.concatenate(c2ws, dtype=np.float32)
        world_view_transforms = np.concatenate(world_view_transforms, dtype=np.float32)
        Ks = np.concatenate(Ks, dtype=np.float32)


        return H, W, Ks, world_view_transforms, file_names, c2ws

    @staticmethod
    def load_pose(pose_file):
        try:
            H, W, K, world_view_transforms, file_names, c2ws = visual_open3d.read_json(pose_file)
        except:
            H, W, K, world_view_transforms, file_names, c2ws = visual_open3d.read_muti_camera_json(pose_file)
            K = K[0]
        return H, W, K, world_view_transforms, file_names, c2ws

if __name__ == '__main__':
    OPEN3D_ARKit_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])  # iOS ARKit uses right-hand coordinate system

    parser = argparse.ArgumentParser()
    # polycam pose
    parser.add_argument("--transform_file_sub_1", type=str, default='/Users/binze/Desktop/scene_split/sfm_test/block_0/transforms_opensfm_1.json')
    # colmap pose
    parser.add_argument("--transform_file_sub_2", type=str,
                        default='/Users/binze/Desktop/scene_split/sfm_test/block_1/transforms_opensfm_1.json')

    parser.add_argument("--ply_polycam", type=str, default='points3d.ply')
    parser.add_argument("--ply_colmap", type=str, default='points3D.ply')
    args = parser.parse_args()

    transform_file_sub_1 = Path(args.transform_file_sub_1)
    transform_file_sub_2 = Path(args.transform_file_sub_2)
    pcd_path_1 = str(transform_file_sub_1.parent.joinpath('point_cloud', 'iteration_30000', 'point_cloud.ply'))
    pcd_path_2 = str(transform_file_sub_2.parent.joinpath('point_cloud', 'iteration_30000', 'point_cloud.ply'))

    # load mesh or ponits
    scene = []
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(pcd_path_1)#.transform(OPEN3D_ARKit_transform)
    scene.append(pcd)
    # 读取相机pose并将其表示为四凌锥
    H, W, K, world_view_transforms, file_names = visual_open3d.load_pose(transform_file_sub_1)
    for i in range(0, world_view_transforms.shape[0]):
    # for i in range(0, 2):
        intrinsic = K
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = world_view_transforms[i][:3, :3]
        extrinsic[:3, 3] = np.linalg.inv(world_view_transforms[i])[-1, :3]

        # cam = create_camera_frustum(intrinsic, extrinsic @ OPEN3D_ARKit_transform, color=[0, 1, 0], img_path=str(transform_file_baseline.parent.joinpath('images_anythingsplit', file_names[i])))
        cam = visual_open3d.create_camera_frustum(intrinsic, extrinsic @ OPEN3D_ARKit_transform, color=[1, 0, 0])
        scene.extend(cam)
    #
    H, W, K, world_view_transforms, file_names = visual_open3d.load_pose(transform_file_sub_2)
    for i in range(0, world_view_transforms.shape[0]):
    # for i in range(0, 2):
        intrinsic = K
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = world_view_transforms[i][:3, :3]
        extrinsic[:3, 3] = np.linalg.inv(world_view_transforms[i])[-1, :3]
        cam = visual_open3d.create_camera_frustum(intrinsic, extrinsic@ OPEN3D_ARKit_transform, color=[0, 1, 0])
        scene.extend(cam)

    o3d.visualization.draw_geometries(scene)

