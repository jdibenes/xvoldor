
import numpy as np
import xv_file

blob = np.load('dav3_poses.npz')
sequence = 'hl2_5'
poses_gt_file = xv_file.scan_files(f'./data/{sequence}/pose')
hl2_to_opencv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)

#NpzFile 'dav3_poses.npz' with keys: extrinsics, intrinsics, depth, conf
#blob['extrinsics'].shape : (195, 3, 4)
print(blob['extrinsics'].shape)


error_r = 0
error_t = 0
count = 0

for key in range(0, 195):
    i = int(key)
    pose_gt = np.fromfile(poses_gt_file[i],dtype=np.float32).reshape((4, 4)).T
    if (count == 0):
        pose_ref = pose_gt
        global_scale = 1.0
    rel_pose_gt = hl2_to_opencv @ (np.linalg.inv(pose_ref) @ pose_gt) @ hl2_to_opencv
    pose_vo = np.vstack((blob['extrinsics'][i, :, :], np.array([0, 0, 0, 1], dtype=np.float32)))
    #pose_vo = np.linalg.inv(pose_vo)
    if (count == 0):
        pose_vo_ref = pose_vo
    pose_vo = pose_vo @ np.linalg.inv(pose_vo_ref)
    if (count >= 1):
        hl2_scale = np.linalg.norm(rel_pose_gt[0:3, 3])
        vggt_scale = np.linalg.norm(pose_vo[0:3, 3])
        global_scale = hl2_scale / vggt_scale    
    pose_vo[0:3, 3] = global_scale * pose_vo[0:3, 3]

    pose_error = pose_vo @ rel_pose_gt

    
    print(pose_vo)
    print(rel_pose_gt)
    print(pose_error)
    
    #if (count >= 1):
        #print(pose_vo[0:3, 3]/ np.linalg.norm(pose_vo[0:3, 3]))
        #print(rel_pose_gt[0:3, 3]/np.linalg.norm(rel_pose_gt[0:3, 3]))
        #quit()

    #print('POSES')
    #print(rel_pose_gt)
    #print(np.linalg.inv(pose_vo))
    print(i)

    error_r += np.arccos(np.clip((np.trace(pose_error[:3, :3]) - 1) / 2, -1, 1))
    error_t += np.linalg.norm(pose_error[:3, 3])
    count += 1

# avg error
print(np.rad2deg(error_r) / count)
print(error_t / count)
print(count)
