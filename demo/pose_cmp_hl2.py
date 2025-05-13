
import numpy as np
import kitti
import xv_file
import os

sequence = 'hl2_6'
mode = 'stereo'
toolname = 'gt'
fname_cmp = os.path.join('./poses', f'pose_{sequence}_{mode}_{toolname}.txt')

poses_cmp_obj = kitti.pose_loader(fname_cmp)
poses_gt_file = xv_file.scan_files(f'./data/{sequence}/pose')

print(poses_cmp_obj.count())
print(len(poses_gt_file))

hl2_to_opencv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)

error_r = 0
error_t = 0

for i in range(0, len(poses_gt_file)):
    pose_gt = np.fromfile(poses_gt_file[i],dtype=np.float32).reshape((4, 4)).T
    if (i == 0):
        pose_ref = pose_gt
    rel_pose_gt = hl2_to_opencv @ (np.linalg.inv(pose_ref) @ pose_gt) @ hl2_to_opencv
    pose_vo = poses_cmp_obj.get_pose(i)
    pose_error = np.linalg.inv(pose_vo) @ rel_pose_gt

    error_r += np.arccos(np.clip((np.trace(pose_error[:3, :3]) - 1) / 2, -1, 1))
    error_t += np.linalg.norm(pose_error[:3, 3])


    print('################')
    print(rel_pose_gt)
    print(pose_vo)    
    print(pose_error)

# avg error
print(np.rad2deg(error_r) / poses_cmp_obj.count()) # 0.009222941243538691 deg
print(error_t / poses_cmp_obj.count()) # 0.006117813473687373 m
