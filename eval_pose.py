import torch
import scipy
import numpy as np
from pathlib import Path
from sfm_free_utils.colmap_2_json import colmap_to_json
from sfm_free_utils.utils import load_pose
from sfm_free_utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from sfm_free_utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from sfm_free_utils.vis_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose

class Eval_pose:

    def __init__(self, root_gt, root_pred):
        self.root_gt = root_gt
        self.root_pred = root_pred # colamp format

    def read_pose(self):

        _, _, _, _, file_names_pred, c2ws_pred = load_pose(self.root_pred)

        # 构建字典
        query_pred = {k:v for k,v in zip(file_names_pred, c2ws_pred)}

        # trans colmap to json pose
        root_file = colmap_to_json(Path(self.root_gt), output_dir=Path(self.root_pred).parent)
        _, _, _, _, file_names_root, c2ws_root = load_pose(root_file)

        # alignment 
        c2ws_pred_align = []
        for i in file_names_root:
            c2ws_pred_align.append(query_pred[i][None])
        c2ws_pred_align = np.concatenate(c2ws_pred_align)

        return c2ws_root, c2ws_pred_align

    def align_pose(self, pose1, pose2):
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R

    def eval_pose(self, poses_gt, poses_pred):
        poses_gt, poses_pred = torch.from_numpy(poses_gt), torch.from_numpy(poses_pred)
        # align scale first (we do this because scale differennt a lot)
        trans_gt_align, trans_est_align, _ = self.align_pose(poses_gt[:, :3, -1].numpy(),
                                                             poses_pred[:, :3, -1].numpy())
        poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
        poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

        c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)
        ate = compute_ATE(poses_gt.cpu().numpy(),
                          c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),
              '&' "{0:.3f}".format(rpe_rot * 180 / np.pi),
              '&', "{0:.3f}".format(ate))
        plot_pose(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy(), Path(self.root_pred))
        with open(f"{Path(self.root_pred).parent}/pose_eval.txt", 'w') as f:
            f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
            f.close()

    def __call__(self):

        c2ws_root, c2ws_pred_align = self.read_pose()
        self.eval_pose(c2ws_root, c2ws_pred_align)


# root_gt = "/home/guozebin/work_code/sfm-free-gaussian-splatting/data/Tanks/Barn/colmap/sparse/0"
# root_pred = "/home/guozebin/work_code/sfm-free-gaussian-splatting/output/Tanks_0727/Barn_coarse_1/transforms.json"

# Evaler = Eval_pose(root_gt, root_pred)
# Evaler()


import os

Tanks_sourse = '/home/guozebin/work_code/sfm-free-gaussian-splatting/data/Tanks'
Tanks_model_path = '/home/guozebin/work_code/sfm-free-gaussian-splatting/output/Tanks_0727'
for scene in sorted(os.listdir(Tanks_model_path)):
    print(scene)
    if 'Horse' not in scene: continue
    root_gt = os.path.join(Tanks_sourse, scene.split('_')[0], "colmap/sparse/0/")
    root_pred = os.path.join(Tanks_model_path, scene, "transforms.json")

    Evaler = Eval_pose(root_gt, root_pred)
    Evaler()