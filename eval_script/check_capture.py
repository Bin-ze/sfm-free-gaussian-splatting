#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   check_capture.py
@Time    :   2024/04/02 14:31:43
@Author  :   Bin-ze 
@Version :   1.0
@Desc    :  Description of the script or module goes here. 
'''

import math
import cv2
from pathlib import Path
from tqdm import tqdm
import os

def save_vidio_folder(PATH, video_path = f'test.mp4'):
    count = 0
    imgs_pred = sorted(list(Path(PATH).glob('*jpg')))
    for img_pred in tqdm(imgs_pred, desc="composite video"):
        if "_image" not in str(img_pred): continue
        count += 1
        im = cv2.imread(str(img_pred))
        if count == 1:
            fps, w, h = 10, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im)
    print('Done!')


# Tanks_sourse = '/home/guozebin/work_code/sfm-free-gaussian-splatting/data/Tanks'

# scene = [i for i in os.listdir(Tanks_sourse)]

# for i in scene:
#     path = f'/home/guozebin/work_code/sfm-free-gaussian-splatting/output/Tanks/{i}/camera_tracking'

#     save_vidio_folder(path, video_path=f'{i}.mp4')

path = '/home/guozebin/work_code/sfm-free-gaussian-splatting/output/demo5/camera_tracking'
save_vidio_folder(path, video_path=f'demo5.mp4')