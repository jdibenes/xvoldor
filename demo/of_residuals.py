
import numpy as np
import os
import xv_file
import xv_flow

path_in = './data/MPI-Sintel/training'
path_gt = os.path.join(path_in, 'flow_GT')



for sequence in ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4', 'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']:
    files_gt = sorted(xv_file.scan_files(os.path.join(path_gt, sequence)))
    for model in ['searaft', 'ptl-memflow', 'ptl-neuflow2', 'ptl-pwcnet', 'ptl-maskflownet']:
        path_fl = os.path.join(path_in, f'flow_{model}')
        files_fl = sorted(xv_file.scan_files(os.path.join(path_fl, sequence)))
        path_res = os.path.join(path_in, f'res_{model}', sequence)
        os.makedirs(path_res, exist_ok=True)
        for file_gt, file_fl in zip(files_gt, files_fl):
            flow_gt = xv_flow.flo_to_flow(file_gt)
            flow_fl = xv_flow.flo_to_flow(file_fl)

            diff = flow_gt - flow_fl

            res = np.linalg.norm(diff, axis=2)
            mag = np.linalg.norm(flow_gt, axis=2)

            tag = np.stack((res, mag), axis=2)

            _, name_gt, _ = xv_file.get_file_name(file_gt)

            fname_tag = os.path.join(path_res, f'{name_gt}.flo')
            xv_flow.flow_to_flo(tag, fname_tag)
            print(f'wrote {fname_tag}')





            #print(flow_gt[0,0,:])
            #print(flow_fl[0,0,:])
            #print(res[0,0])
            #print(mag[0,0])
            #print(tag[0,0,:])

            #quit()









