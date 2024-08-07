# sfm-free-gaussian-splatting
This is a personal project to implement gaussian-splatting without using sfm
## TODO
- [x] Implementing visual odometry based on dust3r for coarse estimating camera pose
- [x] Implementing progressive 3DGS mapping using dust3r(mast3r) keyframe points cloud
- [x] Implementation of fine adjustment of camera pose based on photometric error
- [x] Design training strategies to optimize both camera tracking and incremental mapping
- [ ] Interactive design: visual training process and map display
- [ ] Added currently open source robust 3dgs optimization methods to improve scene representation

## Comparative Test with colmap-free 3d-gs in Tanks Dataset
| Tanks scenes | focal |
| --- | --- | 
|Horse |592.3|
|Ballroom|582.94|
|Barn|593.48|
|church|588.91|
|Family| 587.95|
|Francis|597.12|
|Ignatius|593.98|
|Museum|593.30|

## get start

### Environment Configuration

Please refer to [mast3r](https://github.com/naver/mast3r) and [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### run

```
CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Ballroom -m output/Tanks_0727/Ballroom_coarse_1 --focal_known 582.94 --local_scene_iterations 300 --using_focal --port 6033 

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Barn -m output/Tanks_0727/Barn_coarse_1  --focal_known 593.48 --local_scene_iterations 300 --using_focal --port 6033

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Church -m output/Tanks_0727/Church_coarse_1  --focal_known 588.91 --local_scene_iterations 300 --using_focal --port 6033

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Family -m output/Tanks_0727/Family_coarse_1  --focal_known 587.95 --local_scene_iterations 300 --using_focal --port 6033

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Francis -m output/Tanks_0727/Francis_coarse_1  --focal_known 597.12 --local_scene_iterations 300 --using_focal --port 6033

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Ignatius -m output/Tanks_0727/Ignatius_coarse_1  --focal_known 593.98 --local_scene_iterations 300 --using_focal --port 6033

CUDA_VISIBLE_DEVICES=0 python train_sfmfree.py -s data/Tanks/Museum -m output/Tanks_0727/Museum_coarse_1  --focal_known 593.30 --local_scene_iterations 300 --using_focal --port 6033

```

### low Resolution -r2
| scenes | PSNR | SSIM | LPIPS | RPEt | RPEr | ATE |
| --- | --- | --- | --- | --- | --- | --- |
| church | 32.4852142 | 0.9503435 | 0.0443851 | 0.009 | 0.046 | 0.001 |
| Barn | 35.0659218 | 0.9451419 | 0.0636507 | 0.045 | 0.046 | 0.007 |
| Museum | 37.3292732 | 0.9826216 | 0.0162066 | 0.017 | 0.028 | 0.001 |
| Family | 36.1340218 | 0.9812867 | 0.0189178 | 0.011 | 0.025 | 0.000 |
| Horse | 37.1605682 | 0.9828219 | 0.0187860 | 0.082 | 0.028 | 0.001 |
| Ballroom | 37.9369087 | 0.9873584 | 0.0110291 | 0.014 | 0.021 | 0.000 |
| Francis | 38.3187752 | 0.9728101 | 0.0754065 | 0.007 | 0.025 | 0.001 |
| Ignatius | 34.6665268 | 0.9684885 | 0.0369985 | 0.294 | 0.424 | 0.014 |

### High Resolution
| scenes | PSNR | SSIM | LPIPS | RPEt | RPEr | ATE |
| --- | --- | --- | --- | --- | --- | --- |
| church | 30.0561676 | 0.9239016| 0.1009124  | 0.012  | 0.073 | 0.001 |
| Barn | 31.8307953 | 0.8970286 | 0.1298204 | 0.038 | 0.038 | 0.006 |
| Museum | 34.1565781  | 0.9601846 | 0.0581100 | 0.017 | 0.023 | 0.001 |
| Family | 33.1006699  | 0.9546808 | 0.0688388  | 0.011 | 0.023 | 0.000 |
| Horse | 33.5351334  | 0.9618059  | 0.0543353  | 0.081  | 0.028 | 0.001 |
| Ballroom | 34.6704597  | 0.9710099   | 0.0356095  | 0.033 |0.094 | 0.001 |
| Francis | 34.0897827   | 0.9321958  | 0.1348356 | 0.007 | 0.022 | 0.001 |
| Ignatius | 29.2164097 | 0.9224545  | 0.0969569 | 0.168 | 0.258 | 0.009 |


## Reference Implementation
[dust3r](https://github.com/naver/dust3r)

[mast3r](https://github.com/naver/mast3r)

[CF-3DGS](https://github.com/NVlabs/CF-3DGS)

[wild-gaussian-splatting](https://github.com/nerlfield/wild-gaussian-splatting)

[GaussianSplattingSLAM](https://rmurai.co.uk/projects/GaussianSplattingSLAM/)