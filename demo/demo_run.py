
import os

big_table = [
    ('KITTI-00', 359.428,   359.428,   303.5964,  92.60785, 191.334975, 0.5, 0.25), # 0
    ('KITTI-01', 359.428,   359.428,   303.5964,  92.60785, 191.334975, 0.5, 0.25), # 1
    ('KITTI-02', 359.428,   359.428,   303.5964,  92.60785, 191.334975, 0.5, 0.25), # 2
    ('KITTI-03', 360.76885, 360.76885, 304.77965, 86.427,   192.19074,  0.5, 0.25), # 3
    ('KITTI-04', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 4
    ('KITTI-05', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 5
    ('KITTI-06', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 6
    ('KITTI-07', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 7
    ('KITTI-08', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 8
    ('KITTI-09', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 9
    ('KITTI-10', 353.5456,  353.5456,  300.94365, 91.5552,  190.173765, 0.5, 0.25), # 10
]

if __name__ == '__main__':
    sequence_index = 10
    toolset = 'ptl-neuflow2'
    mode_name = 'stereo'
    set_save_pose = True
    set_enable_mapping = True
    set_enable_loop_closure = True    

    sequence, fx_val, fy_val, cx_val, cy_val, bf_val, resize_val, abs_resize_val = big_table[sequence_index]

    path_base = 'E:/voldor_data/data'#'./data'
    pose_base = './poses'
    path_flow = os.path.join(path_base, sequence, f'flow_{toolset}')
    path_disp = os.path.join(path_base, sequence, f'disp_{toolset}')
    path_img = os.path.join(path_base, sequence, 'img')
    fname_pose = os.path.join(pose_base, f'pose_{sequence}_{mode_name}_{toolset}.txt')

    cmd = 'C:/Users/jcds/AppData/Local/Programs/Python/Python36/python.exe c:/Users/jcds/Documents/GitHub/xvoldor/demo/demo.py'
    fx = f'--fx {fx_val}'
    fy = f'--fy {fy_val}'
    cx = f'--cx {cx_val}'
    cy = f'--cy {cy_val}'
    bf = f'--bf {bf_val}'
    flow_dir = f'--flow_dir {path_flow}'
    img_dir = f'--img_dir {path_img}'
    disp_dir = f'--disp_dir {path_disp}'
    mode = f'--mode {mode_name}'
    enable_mapping = '--enable_mapping' if (set_enable_mapping) else ''
    enable_loop_closure = '--enable_loop_closure ./ORBvoc.bin' if (set_enable_loop_closure) else ''
    resize = f'--resize {resize_val}'
    abs_resize = f'--abs_resize {abs_resize_val}'
    save_pose = f'--save_pose {fname_pose}' if (set_save_pose) else ''

    os.system(f'{cmd} {fx} {fy} {cx} {cy} {bf} {flow_dir} {img_dir} {disp_dir} {mode} {enable_mapping} {enable_loop_closure} {resize} {abs_resize} {save_pose}')
