
import numpy as np
import os
import xv_file
import xv_flow


sequence = 'hl2_5'
toolset = 'searaft'
max_error = 10



path_base = './data'
path_flow_1 = os.path.join(path_base, sequence, f'flow_{toolset}')
path_flow_2 = os.path.join(path_base, sequence, f'flow_2_{toolset}')
path_out = os.path.join(path_base, sequence, f'flow_{toolset}_masked')

os.makedirs(path_out, exist_ok=True)

flows_1 = sorted(xv_file.scan_files(path_flow_1))
flows_2 = sorted(xv_file.scan_files(path_flow_2))

sets = zip(flows_1[:-1], flows_1[1:], flows_2)

for filename_01, filename_12, filename_02 in sets:
    flow_01 = xv_flow.flo_to_flow(filename_01)
    flow_12 = xv_flow.flo_to_flow(filename_12)
    flow_02 = xv_flow.flo_to_flow(filename_02)

    error_flow = np.linalg.norm(flow_01 + flow_12 - flow_02, axis=2)
    mask = error_flow > max_error
    #print(mask.shape)

    flow_01 = flow_01.copy()
    #print(flow_01.shape)
    flow_01[mask, :] = 0

    _, name, ext = xv_file.get_file_name(filename_01)
    xv_flow.flow_to_flo(flow_01, os.path.join(path_out, name + ext))
    #break

flow_01 = xv_flow.flo_to_flow(flows_1[-1])
_, name, ext = xv_file.get_file_name(flows_1[-1])
xv_flow.flow_to_flo(flow_01, os.path.join(path_out, name + ext))
