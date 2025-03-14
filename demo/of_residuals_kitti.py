
import numpy as np
import os
import xv_file
import xv_flow
import cv2

path_in = './data/kitti-flow/training'
path_gt = os.path.join(path_in, 'flow_occ')

files_gt = sorted(xv_file.scan_files(path_gt))

for model in ['searaft', 'ptl-memflow', 'ptl-neuflow2', 'ptl-pwcnet', 'ptl-maskflownet']:
    path_fl = os.path.join(path_in, f'flow_{model}')
    files_fl = sorted(xv_file.scan_files(path_fl))

    path_res = os.path.join(path_in, f'res_{model}')
    os.makedirs(path_res, exist_ok=True)

    for file_gt, file_fl in zip(files_gt, files_fl):
        flow_gt = cv2.imread(file_gt, cv2.IMREAD_UNCHANGED)
        flow_gt = flow_gt[:,:,::-1]

        #print(flow_gt[180:182, 620:622, :])

        invalid = (flow_gt[:, :, 2] <= 0)
        flow_gt = (flow_gt[:, :, 0:2].astype(np.float32) - 32768.0) / 64.0

        flow_fl = xv_flow.flo_to_flow(file_fl)
        #flow_gt = xv_flow.flo_to_flow(file_gt)

        diff = flow_gt - flow_fl

        res = np.linalg.norm(diff, axis=2)
        mag = np.linalg.norm(flow_gt, axis=2)
        mag[invalid] = -1.0

        tag = np.stack((res, mag), axis=2)

        _, name_gt, _ = xv_file.get_file_name(file_gt)

        fname_tag = os.path.join(path_res, f'{name_gt}.flo')
        xv_flow.flow_to_flo(tag, fname_tag)
        print(f'wrote {fname_tag}')

        #print(flow_fl[180:182, 620:622, :])
        #print(flow_gt[180:182, 620:622, :])
        #print(res[180:182, 620:622])
        #print(mag[180:182, 620:622])
        #print(valid[180:182, 620:622])
        #quit()
