#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.append('./mast3r/dust3r/')
sys.path.append('./mast3r/')
sys.path.append('./gaussian-splatting/')
from random import sample
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule
from dust3r.optim_factory import adjust_learning_rate_by_lr

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_scale_regularization_loss, depth_loss_dpt
from gaussian_renderer import render, network_gui
import sys
from scene import Scene_Free, GaussianModel_SFMFree
from utils.general_utils import safe_state
from utils.pose_estimation_utils import save_pose
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, colorize
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import open3d as o3d
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def debug_gs_delate(xyz, mask, save_path='1.ply'):

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz[~mask].cpu().numpy())
    o3d.io.write_point_cloud(save_path, pcd_o3d)

def delate_step(gaussians, viewpoint_cam, step=None):
    '''
    为了减缓重建过程中由于深度估计错误导致的过拟合，从而严重影响定位精度，在重定位之前，我们需要根据深度裁剪场景 
    '''
    gt_depth = viewpoint_cam.depth.cuda().clone().squeeze()
    # 现在，把可见点投影到2d平面
    xyz = gaussians.get_xyz.data
    suffix = torch.ones(xyz.shape[0], device=xyz.device)[:, None]
    xyz_homogeneous = torch.hstack([xyz, suffix])

    pixel_xyz = (viewpoint_cam.camrea_matrix @ xyz_homogeneous.T).T
    pixel_xy = pixel_xyz[:, :2] / pixel_xyz[:, -1].unsqueeze(1)
    mask = torch.all(torch.cat([pixel_xy[:, 0][None] < viewpoint_cam.image_width, pixel_xy[:, 1][None] < viewpoint_cam.image_height, pixel_xy[:, 1][None] > 0, pixel_xy[:, 0][None] >0]), 0)
    
    delate_mask = torch.zeros_like(mask)
    #重置越界点云

    forward_pixel_con = pixel_xy[mask].type(torch.long)
    # 计算gt depth 与 xyz点云的depth的误差
    x_coords = forward_pixel_con[:, 0]
    y_coords = forward_pixel_con[:, 1]
    # 深度测试
    z_no_test = pixel_xyz[:, -1][mask]

    depth = torch.full(gt_depth.shape, float('inf')).cuda()
    
    for i in range(x_coords.shape[0]):
        x, y = x_coords[i], y_coords[i]
        depth[y, x] = min(depth[y, x], z_no_test[i])

    error = torch.abs(gt_depth[y_coords, x_coords] - depth[y_coords, x_coords])
    # thr = 1
    mask_error = error > 3
    delate_mask[mask] = mask_error
    print(f"delate {(delate_mask).sum()} points")
    gaussians.prune(delate_mask)
    if step is not None:
        debug_gs_delate(xyz, delate_mask, save_path=f'{step}.ply')

def local_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_cam, schedule='cosine', lr_base=5e-3, lr_min=1e-5):# 新加入场景的相机只优化相机姿态，而不改变任何其他属性
    # 为什么依靠光度误差进行姿态调整会导致不稳定？
    # 怎样的姿态优化方式可以提供稳定的camear tracking
    first_iter = 0
    print(f"init local cam: {viewpoint_cam.world_view_transform.T.inverse()}")
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.local_scene_iterations), desc="local_train_loop progress")
    first_iter += 1

    if opt.coarse_to_fine:
        gt_depth = F.interpolate(viewpoint_cam.depth.cuda().clone().unsqueeze(0), scale_factor=0.5)[0].clone()
        gt_image = F.interpolate(viewpoint_cam.original_image.cuda().clone().unsqueeze(0), scale_factor=0.5)[0].clone()

    for iteration in range(first_iter, opt.local_scene_iterations + 1):  
        # test     
        t = iteration / opt.local_scene_iterations
        if schedule == 'cosine':
            lr = cosine_schedule(t, lr_base, lr_min)
        elif schedule == 'linear':
            lr = linear_schedule(t, lr_base, lr_min)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if opt.coarse_to_fine and iteration < opt.local_scene_iterations // 2: # 定位时使用低分辨率图像
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, h=viewpoint_cam.image_height // 2, w=viewpoint_cam.image_width // 2)
        else:
            gt_depth = viewpoint_cam.depth.cuda().clone()
            gt_image = viewpoint_cam.original_image.cuda().clone()
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        # Loss 
        # 策略一：选取稳健的像素区域进行光度误差优化
        if opt.using_trusted_pixels:
            gt_depth_ = gt_depth / (gt_depth.max() + 1e-5)
            # 稳健的相机姿态估计需要的是值得信赖的像素区域，因此有必要通过渲染深度和提供深度的差异，确定用于学习的像素区域
            mask = torch.where(torch.abs(gt_depth_ - depth) > opt.trusted_pixels_thr, 0, 1)
            image *= mask
            depth *= mask
            # 需要对3dgs进行裁剪，当发现深度差异较大的结果时
            # torchvision.utils.save_image(torch.hstack([gt_image.cpu(), image.cpu()]), f"{viewpoint_cam.image_name.replace('.', '_gt_pred_image_local.')}")
            Ll1 = l1_loss(image, gt_image * mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image * mask))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                            "totol points": f"{scene.gaussians.get_xyz.shape[0]}",
                            "local view lr": f"{lr:.{7}f}",
                            "cam_rot_delta lr": viewpoint_cam.optimizer.param_groups[0]['lr'],
                            "cam_trans_delta lr": viewpoint_cam.optimizer.param_groups[1]['lr']
                            })
                progress_bar.update(10)

            if iteration == opt.local_scene_iterations:
                progress_bar.close()

            if viewpoint_cam.pose_adjustment:
                # 学习率策略
                adjust_learning_rate_by_lr(viewpoint_cam.optimizer, lr)
                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad(set_to_none=True)
                # update 
                viewpoint_cam.update_pose()

    print(f"optimizer local cam: {viewpoint_cam.world_view_transform.T.inverse()}")
    with torch.no_grad():
        adjust_learning_rate_by_lr(viewpoint_cam.optimizer, lr_base * 0.2)
        training_report_psnr(iteration, l1_loss, scene, render, (pipe, background), [viewpoint_cam], render_depth=True, mode='camera_tracking')         
    return viewpoint_cam

def progressive_global_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_stack_global, progressive_train_iterations, expend=False, tb_writer=None):# 全局
    first_iter = 0
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ema_loss_for_log = 0.0
    
    # set iteration
    if len(viewpoint_stack_global) == 2: # 第一帧至关重要，如果无法获得好的几何，那么后续优化失败
        progressive_global_train_iterations = 1000
    else:
        progressive_global_train_iterations =  min(max(opt.progressive_global_train_iterations * len(viewpoint_stack_global), 500), 1000) # 希望更加侧重新加入的图像，因为之前的图像已经被优化了很多次
    
    progress_bar = tqdm(range(first_iter, progressive_global_train_iterations), desc="progressive_global_train_loop progress")
    first_iter += 1
    viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))]
    
    if opt.using_focal:
       viewpoint_stack_global_idxs = sample([ i for i in range(len(viewpoint_stack_global))], min(len(viewpoint_stack_global), 20)) + [len(viewpoint_stack_global) - 1] # 只关注后面的几帧， 但是这样可能会伤害前面的场景表示 
    if expend:
       viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))][:10]

    for iteration in range(first_iter, progressive_global_train_iterations + 1):      
        iter_start.record()  
        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        if len(viewpoint_stack_global_idxs) == 0:
            if not opt.using_focal:
                viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))]
            elif expend:
                viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))][:5]
            else:
                viewpoint_stack_global_idxs = sample([ i for i in range(len(viewpoint_stack_global))], min(len(viewpoint_stack_global), 20)) + [len(viewpoint_stack_global) - 1] # 只关注后面的几帧， 但是这样可能会伤害前面的场景表示

        viewpoint_cam_global_idxs = viewpoint_stack_global_idxs.pop(randint(0, len(viewpoint_stack_global_idxs)-1))
        viewpoint_cam = viewpoint_stack_global[viewpoint_cam_global_idxs]
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.depth is not None and dataset.using_depth_progressive:
            gt_depth = viewpoint_cam.depth.cuda()
            gt_depth = gt_depth / (gt_depth.max() + 1e-5)
            gt_depth = 1 / gt_depth.clamp(1e-6)
            pred_depth = 1 / depth.clamp(1e-6) # 视差
            # depth_loss = opt.lambda_depth * l1_loss(depth, gt_depth)
            depth_loss = depth_loss_dpt(pred_depth.squeeze(), gt_depth.squeeze())

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if dataset.using_depth_progressive:
            loss += depth_loss

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                            "totol points": f"{scene.gaussians.get_xyz.shape[0]}",
                            "cam_rot_delta lr": viewpoint_cam.optimizer.param_groups[0]['lr'],
                            "cam_trans_delta lr": viewpoint_cam.optimizer.param_groups[1]['lr']
                            })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # Densification
            if iteration < opt.densify_densify_until_iter_progressive:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter_progressive and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, np.array(scene.cameras_extent)[:len(viewpoint_stack_global)].sum(), size_threshold)
                    # 每添加2张图像，进行一次重置
                    if progressive_train_iterations % 2 == 0 and iteration % opt.opacity_reset_interval_progressive == 0:
                        print("opacity_reset")
                        gaussians.reset_opacity()
                        # 如果密度重置，那么首先应该还在致密化过程中，以完成场景裁剪。其次，应该继续优化，以快速恢复场景密度。

            # Optimizer step
            if iteration < progressive_global_train_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if viewpoint_cam.pose_adjustment:
                    viewpoint_cam.optimizer.step()
                    viewpoint_cam.optimizer.zero_grad(set_to_none=True)
                    # update 
                    viewpoint_cam.update_pose()

    with torch.no_grad():
        training_report_psnr(iteration, l1_loss, scene, render, (pipe, background), viewpoint_stack_global, render_depth=True, mode='', tb_writer=tb_writer)
    

def global_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_stack_global, saving_iterations, testing_iterations, tb_writer):# 全局
    first_iter = 0
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Global progress")
    first_iter += 1
    viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))]
    for iteration in range(first_iter, opt.iterations + 1):      
        iter_start.record()  
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if len(viewpoint_stack_global_idxs) == 0:
           viewpoint_stack_global_idxs = [ i for i in range(len(viewpoint_stack_global))]
        viewpoint_cam_global_idxs = viewpoint_stack_global_idxs.pop(randint(0, len(viewpoint_stack_global_idxs)-1))
        viewpoint_cam = viewpoint_stack_global[viewpoint_cam_global_idxs]
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        if iteration > opt.densify_until_iter:
            scale_regularization_loss = compute_scale_regularization_loss(gaussians, visibility_filter)
        else: 
            scale_regularization_loss = None      
        
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.depth is not None and dataset.using_depth_global:
            gt_depth = viewpoint_cam.depth.cuda()
            gt_depth = gt_depth / (gt_depth.max() + 1e-5)
            gt_depth = 1 / gt_depth.clamp(1e-6)
            pred_depth = 1 / depth.clamp(1e-6) # 视差
            # depth_loss = opt.lambda_rank_depth * l1_loss(pred_depth, gt_depth)
            # depth_loss = opt.lambda_rank_depth * l1_loss(depth, gt_depth)
            depth_loss = depth_loss_dpt(pred_depth.squeeze(), gt_depth.squeeze())

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        if dataset.using_depth_global:
            loss += depth_loss

        if scale_regularization_loss is not None:
            loss += scale_regularization_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                            "totol points": f"{scene.gaussians.get_xyz.shape[0]}"
                            })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), viewpoint_stack_global=viewpoint_stack_global)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, np.array(scene.cameras_extent).sum(), size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if viewpoint_cam.pose_adjustment:
                    viewpoint_cam.optimizer.step()
                    viewpoint_cam.optimizer.zero_grad(set_to_none=True)
                    # update 
                    viewpoint_cam.update_pose()

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(os.path.join(args.model_path, 'camera_tracking'), exist_ok = True)
    os.makedirs(os.path.join(args.model_path, 'progressive_train'), exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene_Free, renderFunc, renderArgs, viewpoint_stack_global):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [viewpoint_stack_global[idx] for idx in range(0, len(viewpoint_stack_global), 8)]}, 
                              {'name': 'train', 'cameras' : viewpoint_stack_global})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 20):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_report_psnr(iteration, l1_loss, scene : Scene_Free, renderFunc, renderArgs, viewpoint_stack_global, render_depth=False, mode='camera_tracking', tb_writer=None):

    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : []}, 
                            {'name': 'train', 'cameras' : viewpoint_stack_global})
    print(' train camera number: ', len(validation_configs[1]['cameras']))
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                render_result  = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                image = torch.clamp(render_result["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                # depth
                if render_depth:
                    depth = render_result["depth"].clone()
                    gt_depth = viewpoint.depth.cuda().clone()
                    gt_depth = gt_depth / (gt_depth.max() + 1e-5)
                    if mode == 'camera_tracking':
                        rend_depth = Image.fromarray(
                                            colorize(torch.hstack([gt_depth.cpu(), depth.cpu()]).numpy(),
                                            cmap='magma_r')).convert("RGB")
                        rend_depth.save(f"{os.path.join(scene.model_path, mode)}/{viewpoint.image_name.replace('.', '_gt_pred_depth.')}")
                        # torchvision.utils.save_image(torch.hstack([gt_depth.cpu(), depth.cpu()]), f"{os.path.join(scene.model_path, mode)}/{viewpoint.image_name.replace('.', '_gt_pred_depth.')}")
                        torchvision.utils.save_image(torch.hstack([gt_image.cpu(), image.cpu()]), f"{os.path.join(scene.model_path, mode)}/{viewpoint.image_name.replace('.', '_gt_pred_image.')}")
                    elif mode == 'progressive_train':
                        torchvision.utils.save_image(torch.hstack([gt_depth.cpu(), depth.cpu()]), f"{os.path.join(scene.model_path, mode)}/{viewpoint.image_name.replace('.', f'_gt_pred_depth_{len(viewpoint_stack_global)}.')}")
                        torchvision.utils.save_image(torch.hstack([gt_image.cpu(), image.cpu()]), f"{os.path.join(scene.model_path, mode)}/{viewpoint.image_name.replace('.', f'_gt_pred_image_{len(viewpoint_stack_global)}.')}")
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
    torch.cuda.empty_cache()
    if tb_writer is not None and mode != 'camera_tracking':
        tb_writer.add_scalar('Progressive_PSNR/local_scene_legth', psnr_test, len(viewpoint_stack_global))

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)  
    gaussians = GaussianModel_SFMFree(dataset.sh_degree)
    scene = Scene_Free(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack_global = scene.getTrainCameras()[0]
    scene.debug_init_scene()

    if dataset.progressive:
        for progressive_train_iterations in range(len(scene.getTrainCameras()) - 1):
            # if len(viewpoint_stack_global) % 2 == 0:
            progressive_global_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_stack_global, progressive_train_iterations+1, tb_writer=tb_writer)
            # 接着选取新的cam viewpoint，并对齐到全局坐标系
            # 坐标系对齐
            local_scene_id = len(viewpoint_stack_global) - 1
            scene.alignment_global_scene_pose(local_scene_id, aliment_keyframe=True)
            local_scene = scene.getTrainCameras()[local_scene_id]
            print("-------------------------------------------")
            print(f"register img {local_scene[-1].image_name}")
            print("-------------------------------------------")
            local_cam = local_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_cam=local_scene[-1])
            #delate step 
            # if len(viewpoint_stack_global) % 4 == 0:
            #     delate_step(opt, pipe, gaussians, background, viewpoint_stack_global[-1])  
            viewpoint_stack_global.append(local_cam)
            
            if (progressive_train_iterations + 1) % opt.add_priori_freq == 0:
                # 每次加入新的cam viewpoint， 考虑添加当前帧的点云到全局场景
                pcd = scene.alignment_global_scene_pcd(local_scene_id, aliment_keyframe=True)
                gaussians.add_local_priori(pcd)
                # delate_step(gaussians, viewpoint_stack_global[-1])  
                # debug self.prior_pcd
                if (progressive_train_iterations + 1) % (opt.add_priori_freq * 5) == 0:
                    o3d.io.write_point_cloud(os.path.join(scene.source_path, f"debug_prior_{progressive_train_iterations}.ply"), scene.prior_pcd)

        progressive_global_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_stack_global, progressive_train_iterations+1, tb_writer=tb_writer)

    else:
        for _ in range(len(scene.getTrainCameras()) - 1):
            local_scene_id = len(viewpoint_stack_global) - 1
            scene.alignment_global_scene_pose(local_scene_id, aliment_keyframe=True)
            local_scene = scene.getTrainCameras()[local_scene_id]
            viewpoint_stack_global.append(local_scene[-1])
        
            # 每次加入新的cam viewpoint， 考虑添加当前帧的点云到全局场景
            # pcd = scene.alignment_global_scene_pcd(local_scene_id, aliment_keyframe=True)
            # gaussians.add_local_priori(pcd)

    # record progressive gs
    print("\n Saving Progressive Gaussians")
    scene.save()
    save_pose(viewpoint_stack_global, save_path=dataset.model_path, r=dataset.resolution, stuff='progressive')
    save_pose(viewpoint_stack_global, save_path=dataset.source_path, r=dataset.resolution, stuff='progressive')
    # refine 
    print("\n init gaussian with refine mode")
    gaussians.training_setup_refine(opt)
    global_train_loop(dataset, opt, pipe, scene, gaussians, background, viewpoint_stack_global, saving_iterations, testing_iterations, tb_writer)
    save_pose(viewpoint_stack_global, save_path=dataset.model_path, r=dataset.resolution)
    save_pose(viewpoint_stack_global, save_path=dataset.source_path, r=dataset.resolution)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[300, 1_000, 7_000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
