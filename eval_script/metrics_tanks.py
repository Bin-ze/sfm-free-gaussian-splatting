import os

Tanks_sourse = '/home/guozebin/work_code/sfm-free-gaussian-splatting/data/Tanks'
Tanks_model_path = '/home/guozebin/work_code/sfm-free-gaussian-splatting/output/Tanks_0727'
for scene in os.listdir(Tanks_model_path):
    if 'Horse' not in scene: continue
    s = os.path.join(Tanks_sourse, scene.split('_')[0])
    m = os.path.join(Tanks_model_path, scene)
    os.system(f"python render.py -s {s} -m {m} --skip_train --eval")
    print(scene)
    os.system(f"python metrics.py -m {m}")
