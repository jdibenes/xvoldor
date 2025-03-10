
import numpy as np
import kitti
import os

def do_cmp(mode, toolname):
    bigtable = [
        ('KITTI-00', '00.txt'),
        ('KITTI-01', '01.txt'),
        ('KITTI-02', '02.txt'),
        ('KITTI-03', '03.txt'),
        ('KITTI-04', '04.txt'),
        ('KITTI-05', '05.txt'),
        ('KITTI-06', '06.txt'),
        ('KITTI-07', '07.txt'),
        ('KITTI-08', '08.txt'),
        ('KITTI-09', '09.txt'),
        ('KITTI-10', '10.txt'),
    ]

    err = []

    for sequence, sequence_gt in bigtable:
        fname_gt = os.path.join('./data', sequence, sequence_gt)
        fname_cmp = os.path.join('./poses', f'pose_{sequence}_{mode}_{toolname}.txt')

        poses_gt_obj = kitti.pose_loader(fname_gt)
        poses_cmp_obj = kitti.pose_loader(fname_cmp)

        err_obj = kitti.calcSequenceErrors(poses_gt_obj, poses_cmp_obj)

        err_r_all = 0
        err_t_all = 0
        for _, err_r, err_t, _, _ in err_obj:
            err_r_all += err_r
            err_t_all += err_t
        err_r = err_r_all / len(err_obj)
        err_t = err_t_all / len(err_obj)

        #err = np.array([err_r, err_t], dtype=np.float64)
        err.append([err_r, err_t])

    fname_out = os.path.join('./errors', f'err_{mode}_{toolname}.txt')
    np.savetxt(fname_out, np.array(err, dtype=np.float64).reshape((-1, 2)), delimiter=' ')
    #np.array(err, dtype=np.float64).reshape((-1, 2)).tofile(fname_out, sep=' ')
    #print(err)

for n_mode in ['stereo', 'mono-scaled', 'mono']:
    print(n_mode)
    for n_tool in ['ptl-maskflownet', 'ptl-memflow', 'ptl-neuflow2', 'searaft', 'ptl-pwcnet']:
        do_cmp(n_mode, n_tool)






#fname_gt = './data/KITTI-03/03.txt'
#fname_cmp = './poses/pose_KITTI-03_stereo_searaft.txt'

#poses_gt_obj = kitti.pose_loader(fname_gt)
#poses_cmp_obj = kitti.pose_loader(fname_cmp)

#err_obj = kitti.calcSequenceErrors(poses_gt_obj, poses_cmp_obj)
#err_r_all = 0
#err_t_all = 0
#for _, err_r, err_t, _, _ in err_obj:
#    err_r_all += err_r
#    err_t_all += err_t

##print(err_obj)
#print(err_r_all / len(err_obj))
#print(err_t_all / len(err_obj))

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
