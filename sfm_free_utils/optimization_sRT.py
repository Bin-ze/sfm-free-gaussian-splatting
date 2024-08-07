#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   optimization_sRT.py
@Time    :   2024/04/11 15:08:03
@Author  :   Bin-ze 
@Version :   1.0
@Desc    :  Description of the script or module goes here. 
'''

import sys
import torch
import logging
import torch.nn as nn
from torch.utils.data import Dataset

import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import open3d as o3d
import numpy as np

# 定义一些李群函数用于姿态估计
def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    # T = torch.eye(4, device=device, dtype=dtype)
    # T[:3, :3] = R
    # T[:3, 3] = t
    return R, t


class pcd_Dataset(Dataset):
    def __init__(self, source_pcd_path, target_pcd_path, source_camera_point=None, target_camera_point=None):

        self.source_pcd_path = source_pcd_path
        self.target_pcd_path = target_pcd_path

        # 添加相机主点约束
        self.optimizer_camera_center = (source_camera_point is not None and target_camera_point is not None)
        if self.optimizer_camera_center:
            self.source_camera_point = source_camera_point
            self.target_camera_point = target_camera_point

        self.data = self.prepare_train_data()


    @staticmethod
    def load_pcd(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcs = np.asarray(pcd.points)
        # 构建齐次表示
        suffix = np.ones(pcs.shape[0])[:, None]
        pc_homogeneous = np.hstack([pcs, suffix])

        return pc_homogeneous

    def prepare_train_data(self):

        data = self.load_pcd(self.source_pcd_path)
        label = self.load_pcd(self.target_pcd_path)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, item):

        data = self.data[0][item]
        label = self.data[1][item]

        return data, label

    def __len__(self):
        return len(self.data[0])

    def get_center(self):
        center = torch.zeros(4)
        center[:3] = self.data[0][:, :3].mean(0)

        return center

    @staticmethod
    def collate_fn(batch):

        data, labels = tuple(zip(*batch))

        data = torch.stack(data, dim=0)
        labels = torch.stack(labels, dim=0)
        return data, labels

class Euclidean_distance_loss(nn.Module):
    def __init__(self, weight=None, camera_nums=0):
        super(Euclidean_distance_loss, self).__init__()
        self.weight = weight
        self.camera_nums = camera_nums

    def forward(self, inputs, targets):

        if self.weight is not None:
            weights = torch.ones(inputs.shape[0], dtype=torch.float32)
            weights[-self.camera_nums:] = self.weight
            loss = torch.mean(weights.to(inputs.device) * torch.norm(inputs - targets, dim=1)) 
        else:
            loss = torch.mean(torch.norm(inputs - targets, dim=1)) 

        return loss


# 新的建模方式：使用优化相似变换矩阵的方式直接找到source->target之间的变换
class sRT_Optimizer(nn.Module):
    """
    我们建模下面的描述：
        trans_martix = [ sR, t
                         0,  1]
        target_pcd = trans_martix @ source_pcd 
    其中：
        s，R，t为需要优化的量
        s [3,]
        R [3,]
        t [3,]
        共优化9个变量
    """
    def __init__(self, init_pose=None):
        super(sRT_Optimizer, self).__init__()

        # 尝试建模扰动模型
        self.delta = True
        # 给一个初始化姿态
        if init_pose is not None:
            logging.info("priori init")
            self.R = init_pose[:3, :3].clone().to('cuda')   
            self.T = init_pose[:3, 3].clone().to('cuda')   
        else:
            logging.info("random init")
            self.R = torch.eye(3, device='cuda')
            self.T = torch.zeros(3, device='cuda')

        # 建模扰动因子
        self.rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device='cuda')
        )
        self.trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device='cuda')
        )

        self.scale = nn.Parameter(
            torch.ones(3, requires_grad=True, device='cuda')
        )

    def get_transform(self):
        """
        trans_martix = [ sR, t
                        0, 1]

        """

        trans = torch.eye(4, device=self.R.device)
        tau = torch.cat([self.trans_delta, self.rot_delta], axis=0)
        delta_r, delta_t = SE3_exp(tau)
        
        trans[:3, :3] = self.scale * (delta_r @ self.R)
        trans[:3, 3] = self.T + delta_t

        return trans

    def update_RT(self, R, t):
        self.R = R.to(device=self.R.device)
        self.T = t.to(device=self.T.device)

    def update_trans(self, converged_threshold=1e-4):
        tau = torch.cat([self.trans_delta.data, self.rot_delta.data], axis=0)

        delta_r, delta_t = SE3_exp(tau)
        new_R = delta_r @ self.R
        new_T = self.T + delta_t

        converged = tau.norm() < converged_threshold
        self.update_RT(new_R, new_T)

        self.rot_delta.data.fill_(0)
        self.trans_delta.data.fill_(0)

        return converged

    def forward(self, x):
        
        # print(self.R @ self.R.T)
        trans = self.get_transform()
        y = (trans @ x.T).T

        return y

# 新的建模方式：使用优化相似变换矩阵的方式直接找到source->target之间的变换
class sRT_global_Optimizer(nn.Module):
    """
    我们建模下面的描述：
        trans_martix = [ sR, t
                         0,  1]
        target_pcd = trans_martix @ source_pcd 
    其中：
        s，R，t为需要优化的量
        s [3,]
        R [3,]
        t [3,]
        共优化9个变量
    """
    def __init__(self, init_pose=None):
        super(sRT_global_Optimizer, self).__init__()

        # 尝试建模扰动模型
        self.delta = True
        # 给一个初始化姿态
        if init_pose is not None:
            logging.info("priori init")
            self.R = []
            self.T = []
            self.scale = nn.ParameterList()
            self.rot_delta = nn.ParameterList()
            self.trans_delta = nn.ParameterList()
            for i in init_pose:
                self.scale.append(nn.Parameter(
                                torch.tensor(i[0], requires_grad=True, device='cuda')
                                ))
                self.R.append(torch.tensor(i[1]).to('cuda'))
                self.T.append(torch.tensor(i[2]).to('cuda'))
                # 建模扰动因子
                self.rot_delta.append(nn.Parameter(
                    torch.zeros(3, requires_grad=True, device='cuda')
                ))
                self.trans_delta.append(nn.Parameter(
                    torch.zeros(3, requires_grad=True, device='cuda')
                ))

    def get_transform(self):
        """
        trans_martix = [ sR, t
                        0, 1]

        """

        trans = torch.eye(4, device=self.R[0].device)

        for i in range(len(self.R)):
            trans_tmp = torch.eye(4, device=self.R[0].device)
            tau = torch.cat([self.trans_delta[i], self.rot_delta[i]], axis=0)
            delta_r, delta_t = SE3_exp(tau)
            
            trans_tmp[:3, :3] = self.scale[i] * (delta_r @ self.R[i])
            trans_tmp[:3, 3] = self.T[i] + delta_t

            # 
            trans = trans_tmp @ trans 

        return trans


    def update_RT(self, R, t, i):
        self.R[i] = R.to(device=self.R[i].device)
        self.T[i] = t.to(device=self.T[i].device)

    def update_trans(self, converged_threshold=1e-4):
        for i in range(len(self.trans_delta)):
            tau = torch.cat([self.trans_delta[i].data, self.rot_delta[i].data], axis=0)

            delta_r, delta_t = SE3_exp(tau)
            new_R = delta_r @ self.R[i]
            new_T = self.T[i] + delta_t

            converged = tau.norm() < converged_threshold
            self.update_RT(new_R, new_T, i)

            self.rot_delta[i].data.fill_(0)
            self.trans_delta[i].data.fill_(0)

        return converged

    def forward(self, x):
        
        # print(self.R @ self.R.T)
        trans = self.get_transform()
        y = (trans @ x.T).T

        return y

class Model(nn.Module):
    """
    我们建模以下描述：
        s * x 
        trans_l[:3, 3] =  s * trans_l[:3, 3]
        trans_lg = trans_g @ trans_l^-1
        y = (trans_lg @ (s * x).T).T
    其中：
        s为需要优化的变量
        trans_l与trans_g表示在关键帧在local和global坐标系下的c2w矩阵(已知)
    """

    def __init__(self, global_keyframes_pose, local_keyframes_pose, pcd_center):
        super(Model, self).__init__()

        self.scale = nn.Parameter(
            torch.ones(3, requires_grad=True, device='cuda')
        )
        self.global_keyframes_pose = global_keyframes_pose.to('cuda')
        self.local_keyframes_pose = local_keyframes_pose.to('cuda')
        self.pcd_center = pcd_center.to('cuda')

        # self.trans_matrix = nn.Parameter(
        #     torch.eye(4, requires_grad=True, device='cuda')
        # )
        # self.trans_matrix = torch.eye(4, device='cuda')

    def forward(self, x):
        
        # 新的实现
        # y = x - self.pcd_center
        # tmp = y.clone()
        # tmp[:, :3] = self.scale * y[:, :3]
        # # y[:, :3] = self.scale * y[:, :3]
        # y = tmp + self.pcd_center

        # 不在中心缩放
        tmp = x.clone()
        tmp[:, :3] = self.scale * x[:, :3]
        y = tmp

        tmp_1 = self.local_keyframes_pose.clone()
        tmp_1[:3, 3] = self.scale *  self.local_keyframes_pose[:3, 3]

        # self.local_keyframes_pose[:3, :3] = self.scale *  self.local_keyframes_pose[:3, :3]
        
        trans_pose = self.global_keyframes_pose @ torch.inverse(tmp_1)
        
        y = (trans_pose @ y.T).T

        return y


class Trainer:
    def __init__(self, model, optimizer, dataloder, train_num, val_num, save_path, config):

        self.config = config
        # 解析配置
        self.device = self.config['device']
        self.epochs = self.config["epoch"]
        self.val_frequency = self.config["val_frequency"]
        self.save_path = save_path

        self.train_num = train_num
        self.val_num = val_num
        self.error = 100

        self.train_loader, self.val_loader = dataloder
        self.train_steps = len(self.train_loader)

        # build loss
        self.loss = Euclidean_distance_loss(weight=1.0, \
                                            camera_nums=self.train_loader.dataset.source_camera_point.shape[0])

        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.epochs // 2, gamma=0.1)


    def run_epoch(self, epoch):

        self.model.train()
        self.running_loss = 0
        train_bar = self.train_loader
        for step, data in enumerate(train_bar):
            feature, labels = data
            if self.train_loader.dataset.optimizer_camera_center:
                source_keypoint = self.train_loader.dataset.source_camera_point
                target_keypoint = self.train_loader.dataset.target_camera_point
                feature = torch.cat([feature, source_keypoint])
                labels = torch.cat([labels, target_keypoint])

            outputs = self.model(feature.to(self.device))
            loss = self.loss(outputs[:, :3], labels.to(self.device)[:, :3])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if hasattr(self.model, 'delta'):
                self.model.update_trans()

            # print statistics
            self.running_loss += loss.item()

            if step % 100 == 0:
                train_bar = "lr:{} train epoch[{}/{}] loss:{:.3f}".format(self.scheduler.get_last_lr(), epoch + 1, self.epochs, loss)
                logging.info(train_bar)

    @torch.no_grad()
    def validate(self, epoch):
            self.model.eval()
            errors = 0.0  # accumulate accurate number / epoch

            val_bar = self.val_loader
            for val_data in val_bar:
                val_feature, val_labels = val_data
                outputs = self.model(val_feature.to(self.device))
                # 计算距离
                errors += (F.pairwise_distance(val_labels.to(self.device)[:, :3], outputs[:, :3])).sum()

            val_errors = errors / self.val_num

            logging.info('validation')
            logging.info('[epoch %d] train_loss: %.3f  val_errors: %.3f' %
                  (epoch + 1, self.running_loss / self.train_steps, val_errors))

            if not self.config["save_best_pth_only"]:
                torch.save(self.model.state_dict(), self.save_path + f'epoch_{epoch}.pth')

            if val_errors < self.error:
                self.error = val_errors
                torch.save(self.model.state_dict(), self.save_path + 'best.pth')

    def run(self):

        for epoch in range(self.epochs):

            self.run_epoch(epoch)

            self.scheduler.step()

            if (epoch + 1) % self.val_frequency == 0:

                self.validate(epoch)

                print("scale: ", [i.data.cpu().numpy().tolist() for i in self.model.scale] if isinstance(self.model.scale, nn.ParameterList) else self.model.scale.data.cpu().numpy().tolist())            
                print("R: ", [i.data.cpu().numpy().tolist() for i in self.model.R] if isinstance(self.model.R, list) else self.model.R.data.cpu().numpy())
                print("T: ", [i.data.cpu().numpy().tolist() for i in self.model.T] if isinstance(self.model.R, list) else self.model.T.data.cpu().numpy() )
                print("Trans matrix: ", self.model.get_transform().data.cpu().numpy().tolist())

        

        return self.model.get_transform().data.cpu() #, self.model.R.data.cpu().numpy() 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    source_pcd_path = "/home/guozebin/work_code/dust3r/source.ply"
    target_pcd_path = "/home/guozebin/work_code/dust3r/target.ply"

    # instance dataset
    train_dataset = pcd_Dataset(source_pcd_path, target_pcd_path)
    val_dataset = pcd_Dataset(source_pcd_path, target_pcd_path)
    # compute sample number
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    # define hyper-parameter
    config = dict(
        epoch=50,
        batch_size=40960,
        device='cuda',
        Lr = 0.001,
        model_save_path='./',
        save_best_pth_only=1,
        val_frequency=10)

    # instance loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=4,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['batch_size'],
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4,
                                             collate_fn=val_dataset.collate_fn)
    pcd_center = train_dataset.get_center()
    # instance model
    model = sRT_Optimizer()

    # instance optim
    optimizer = optim.Adam(model.parameters(), config['Lr'])

    # init Trainer
    Trainer = Trainer(model=model, optimizer=optimizer,
                      dataloder=[train_loader, val_loader],
                      train_num=train_num, val_num=val_num,
                      save_path=config['model_save_path'], config=config)
    Trainer.run()

