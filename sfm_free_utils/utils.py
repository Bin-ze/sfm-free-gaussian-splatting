import open3d as o3d
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

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

    world_view_transforms = []
    file_names = []
    c2ws = []
    for idx, i in enumerate(contents['frames']):
        if 'pitch' in i['file_path'].split('/')[-1]: 
            print(f"skip {i['file_path'].split('/')[-1]}")
            continue
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
    # sorted
    
    world_view_transforms = np.concatenate(world_view_transforms, dtype=np.float32)
    c2ws = np.concatenate(c2ws, dtype=np.float32)


    return H, W, K, world_view_transforms, file_names, c2ws

def read_muti_camera_json(transform_file):
    with open(str(transform_file), 'r') as f:
        contents = json.load(f)

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
        c2ws.append(matrix.copy()[None])
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

def load_pose(pose_file):
    try:
        H, W, K, world_view_transforms, file_names, c2ws = read_json(pose_file)
    except:
        H, W, K, world_view_transforms, file_names, c2ws = read_muti_camera_json(pose_file)
        K = K[0]
    return H, W, K, world_view_transforms, file_names, c2ws

def write_transformsfile_muti(H, W, Ks, c2ws, save_path, trans=None, R=None, suffix='local', scale=1):
    data = dict()
    frames = []

    if isinstance(c2ws, str):
        def read_json(transform_file):
            with open(str(transform_file), 'r') as f:
                contents = json.load(f)
            return contents
        for i in read_json(c2ws)['frames']:
            Ks.update({Path(i["file_path"]).name:np.array([[i["fl_x"], 0, i["cx"]],
                                                    [0, i["fl_y"], i["cy"]],
                                                    [0, 0, 1]])})
        c2ws = {Path(i["file_path"]).name:np.array(i["transform_matrix"]) for i in read_json(c2ws)['frames']}

    # 读取图片名以及对应的pose
    for k, v in c2ws.items():
        if trans is not None:
            v[:3, :3] = R @ v[:3, :3]
            v[:, 3] = trans @ v[:, 3]

        v[:3, 3] *= scale
        frame = dict(
            fl_x=float(Ks[k][0][0]),
            fl_y=float(Ks[k][1][1]),
            k1=0,
            k2=0,
            k3=0,
            k4=0,
            p1=0,
            p2=0,
            is_fisheye=False,
            cx=float(Ks[k][0][2]),
            cy=float(Ks[k][1][2]),
            w=W,
            h=H,
            aabb_scale=16,
            file_path= f"images/{k}",
            # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
            transform_matrix= [j.tolist() for j in v]
        )
        frames.append(frame)

    data["frames"] = frames
    if suffix is not None:
        with open(f"{save_path}/transforms-{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    else:
        with open(f"{save_path}/transforms.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    return f"{save_path}/transforms-{suffix}.json"


def merge_trans_file(trans_g, trans_l):

    def read_json(transform_file):
        with open(str(transform_file), 'r') as f:
            contents = json.load(f)
        return contents
    
    frames = []
    g_frames = read_json(trans_g)['frames']
    l_frames = read_json(trans_l)['frames']

    g_file_path = [i['file_path'] for i in g_frames]
    l_file_path = [i['file_path'] for i in l_frames]

    repeat_index = []
    for idx, i in enumerate(l_file_path):
        if i in g_file_path:
            repeat_index.append(idx)

    for i in reversed(repeat_index):
        del l_frames[i]

    frames.extend(g_frames)
    frames.extend(l_frames)

    frames_merge = sorted(frames, key= lambda x: x["file_path"])
    data = dict()
    data["frames"] = frames_merge
    with open(f"{str(Path(trans_g).parent)}/transforms-merge.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return f"{str(Path(trans_g).parent)}/transforms-merge.json"

def get_transform(s, R, T):
    s = np.array(s)
    R = np.array(R)
    T = np.array(T)
    trans = np.eye(4)
    # trans[:3, :3] = s * R
    if s.shape[0] == 1:
        trans[:3, :3] = np.diag(s.repeat(3)) @ R
    else:
        trans[:3, :3] = np.diag(s) @ R
    trans[:3, 3] = T
    return trans

def sample_pcd(point_cloud, num_points=200000):
    point_cloud_array = np.asarray(point_cloud.points)
    point_color_array = np.asarray(point_cloud.colors)

    # Generate random indices
    num_total_points = point_cloud_array.shape[0]
    random_indices = np.random.choice(num_total_points, size=num_points, replace=True)

    # Select sampled points using random indices
    sampled_points = point_cloud_array[random_indices, :]
    sampled_colors = point_color_array[random_indices, :]

    point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    point_cloud.colors = o3d.utility.Vector3dVector(sampled_colors)

    return point_cloud

def merge_pcd(pcd1, pcd2):

    point_cloud_array1 = np.asarray(pcd1.points)
    point_color_array1 = np.asarray(pcd1.colors)

    point_cloud_array2 = np.asarray(pcd2.points)
    point_color_array2 = np.asarray(pcd2.colors)

    pcd_merge = np.concatenate([point_cloud_array1, point_cloud_array2])
    col_merge = np.concatenate([point_color_array1, point_color_array2])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_merge)
    pcd.colors = o3d.utility.Vector3dVector(col_merge)

    return pcd

def pcd_trans(pcd, trans):

    pcd_array = np.asarray(pcd.points)

    suffix = np.ones(pcd_array.shape[0])[:, None]
    pcd_homogeneous = np.hstack([pcd_array, suffix])
    trans_pcd = np.dot(trans, pcd_homogeneous.T).T[:, :3]

    pcd.points = o3d.utility.Vector3dVector(trans_pcd)
    return pcd

def decompose_similarity_matrix_nonuniform(M):
    # 提取旋转矩阵
    A = M[:3, :3]

    # 计算缩放向量s
    s = np.linalg.norm(A, axis=1)

    # 对每一行除以对应的缩放系数
    R = A / s[:, np.newaxis]

    # 提取平移向量t
    t = M[:3, 3]

    return s, R, t

def local_2_global_single(path, scale, R, T):
    scale = scale[::-1]
    R = R[::-1]
    T = T[::-1]
    path = path[::-1]

    trans_s2t = []
    trans_s2t_R = []
    space_len = len(scale) + 1
    for i in range(space_len):
        trans = np.eye(4)
        r = np.eye(3)
        for j in range(i):
            t = get_transform(scale[j], R[j], T[j])
            trans = trans @ t
            r = r @ np.array(R[j])

        trans_s2t.append(trans)
        trans_s2t_R.append(r)


    # export 
    for i in tqdm(range(len(path)), desc="align scene for global..."):
        pose = np.load(path[i].joinpath("cams2world_scale.npy"), allow_pickle=True).item()
        # 读取K
        k = np.load(path[i].joinpath("K.npy"), allow_pickle=True).item()
        # export pose

        # 新的计算方法
        # _, R, _ = decompose_similarity_matrix_nonuniform(trans_s2t[i])

        write_transformsfile_muti(H=1472, W=1472, Ks=k, c2ws=pose, save_path=str(path[i]), trans=trans_s2t[i], R=trans_s2t_R[i], suffix='global-coordinate')
        # export pcd
        pcd = o3d.io.read_point_cloud(str(path[i].joinpath('points3d.ply')))
        pcd = pcd_trans(pcd, trans_s2t[i]) 
        pcd = sample_pcd(pcd, num_points=5000000)
        o3d.io.write_point_cloud(str(path[i].joinpath('global.ply')), pcd)

def local_2_global_no_overlap(path, scale, R, T):
    '''
    该实现用于将一个场景分为一个稀疏的global以及一些密集的local的情况
    '''
    # global 
    pose_g = np.load(path[0].joinpath("cams2world_scale.npy"), allow_pickle=True).item()
    # 读取K
    k_g = np.load(path[0].joinpath("K.npy"), allow_pickle=True).item()
    pose_g = write_transformsfile_muti(H=1472, W=1472, Ks=k_g, c2ws=pose_g, save_path=str(path[0]), trans=None, R=None, suffix='global')
    pcd_g = o3d.io.read_point_cloud(str(path[0].joinpath('points3d_scale.ply')))
    pcd_g = sample_pcd(pcd_g)

    for i in tqdm(range(1, len(path)), desc="merge scene ..."):
        pose_l = np.load(path[i].joinpath("cams2world_scale.npy"), allow_pickle=True).item()
        # 读取K
        k_l = np.load(path[i].joinpath("K.npy"), allow_pickle=True).item()
        pose_l = write_transformsfile_muti(H=1472, W=1472, Ks=k_l, c2ws=pose_l, save_path=str(path[i]), trans=get_transform(scale[i-1], R[i-1], T[i-1]), R=R[i-1], suffix='local')
        pcd_l = o3d.io.read_point_cloud(str(path[i].joinpath('points3d_scale.ply')))
        pcd_l = sample_pcd(pcd_l)
        pcd_l = pcd_trans(pcd_l, get_transform(scale[i-1], R[i-1], T[i-1]))
        pose_g = merge_trans_file(pose_g, pose_l)
        pcd_g = merge_pcd(pcd_l, pcd_g)

    pcd = sample_pcd(pcd_g)
    o3d.io.write_point_cloud("Pcd_Global_Alignment_sample_overlap.ply", pcd)

def local_2_global(path, scale, R, T):
    '''
    该实现用于将一个场景按空间分块，块与块之间存在重叠。要想对齐到全局，将从后往前不断的坐标变换
    '''
    for i in tqdm(range(len(path)-1), desc="merge scene ..."):
        if i == 0:

            pose_l = np.load(path[i].joinpath("cams2world_scale.npy"), allow_pickle=True).item()
            # 读取K
            k_l = np.load(path[i].joinpath("K.npy"), allow_pickle=True).item()
            pose_l = write_transformsfile_muti(H=1472, W=1472, Ks=k_l, c2ws=pose_l, save_path=str(path[i]), trans=get_transform(scale[i], R[i], T[i]), R=R[i], suffix='local')
            # try:
            # pcd_l = o3d.io.read_point_cloud(str(path[i].joinpath('points3d_scale_clean.ply')))
            # except:
            pcd_l = o3d.io.read_point_cloud(str(path[i].joinpath('points3d.ply')))
            pcd_l = sample_pcd(pcd_l)
            pcd_l = pcd_trans(pcd_l, get_transform(scale[i], R[i], T[i]))
        else:
            pcd_l = merge_pcd(pcd_l, pcd_g)
            pcd_l = pcd_trans(pcd_l, get_transform(scale[i], R[i], T[i]))
            #
            pose_l = merge_trans_file(pose_g, pose_l) 
            # 
            # 读取K
            k_l = np.load(path[i].joinpath("K.npy"), allow_pickle=True).item()
            pose_l = write_transformsfile_muti(H=1472, W=1472, Ks=k_l, c2ws=pose_l, save_path=str(path[i]), trans=get_transform(scale[i], R[i], T[i]), R=R[i], suffix='local')

        pose_g = np.load(path[i+1].joinpath("cams2world_scale.npy"), allow_pickle=True).item()
        # 读取K
        k_g = np.load(path[i+1].joinpath("K.npy"), allow_pickle=True).item()
        pose_g = write_transformsfile_muti(H=1472, W=1472, Ks=k_g, c2ws=pose_g, save_path=str(path[i+1]), trans=None, R=None, suffix='global')
        # try:
            # pcd_g = o3d.io.read_point_cloud(str(path[i + 1].joinpath('points3d_scale_clean.ply')))
        # except:
        pcd_g = o3d.io.read_point_cloud(str(path[i + 1].joinpath('points3d.ply')))
        pcd_g = sample_pcd(pcd_g)


    pose_merge = merge_trans_file(pose_g, pose_l)
    print(pose_merge)
    pcd_merge = merge_pcd(pcd_l, pcd_g)
    pcd = sample_pcd(pcd_merge, num_points=500000)
    # o3d.io.write_point_cloud("Pcd_Global_Alignment.ply", pcd_merge)
    o3d.io.write_point_cloud("Pcd_Global_Alignment_sample.ply", pcd)
