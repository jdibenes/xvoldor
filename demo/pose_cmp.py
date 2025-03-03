
import numpy as np
import kitti

fname_gt = './data/KITTI-03/03.txt'
fname_cmp = './poses/pose_KITTI-03_stereo_searaft.txt'

poses_gt_obj = kitti.pose_loader(fname_gt)
poses_cmp_obj = kitti.pose_loader(fname_cmp)

err_obj = kitti.calcSequenceErrors(poses_gt_obj, poses_cmp_obj)
err_r_all = 0
err_t_all = 0
for _, err_r, err_t, _, _ in err_obj:
    err_r_all += err_r
    err_t_all += err_t

#print(err_obj)
print(err_r_all / len(err_obj))
print(err_t_all / len(err_obj))

''' 
KITTI-05
sea-raft
3.963874898377582e-05
0.013400698399582443

mfn
3.700554331620875e-05
0.01365362899784287

KITTI-03
mfn
6.049115391985042e-05
0.03359945168407064

seareaft
0.00013845200331848092
0.035543498202380175
'''
