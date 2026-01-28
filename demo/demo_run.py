
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
    ('hl2_3',    586.27075, 586.6171,  374.04108, 202.26265, 315.3576,  1.0, 0.5), # 11
    ('hl2_3_st', 586.27075, 586.6171,  374.04108, 202.26265, 315.3576,  1.0, 0.5), # 12
    ('hl2_3_vlc', 352.01007, 359.45428, 249.33237, 292.87296, -0.09950941*352.01007, 1.0, 0.5), # 13
    ('c1',        485.94, 499.69, 321.84, 188.49, 96, 1.0, 1.0), # 14 # bf=96
    ('c2',        480.52, 496.01, 321.70, 181.73, 64, 1.0, 0.5), # 15
    ('c3',        479.05, 483.88, 321.21, 183.82, 64, 1.0, 0.5), # 16
    ('c4',        589.51, 595.46, 317.49, 186.80, 64, 1.0, 0.5), # 17
    ('hl2_5', 586.27075, 586.27075,  374.04108, 202.26265, 117.254150390625,  1.0, 1.0), # 18
    ('hl2_6', 586.27075, 586.27075,  374.04108, 202.26265, 117.254150390625,  1.0, 1.0), # 19
    ('hl2_7', 586.27075, 586.27075,  374.04108, 202.26265, 117.254150390625,  1.0, 1.0), # 20
]

'''
[[352.01007   0.        0.        0.     ] 
 [  0.      359.45428   0.        0.     ] 
 [249.33237 292.87296   1.        0.     ] 
 [  0.        0.        0.        1.     ]]
[[356.10004   0.        0.        0.     ]
 [  0.      360.17044   0.        0.     ]
 [238.04337 301.9479    1.        0.     ]
 [  0.        0.        0.        1.     ]]        
[[ 0.99905    -0.03802227  0.02129362  0.        ] 
 [ 0.03747033  0.99896604  0.02574591  0.        ] 
 [-0.02225052 -0.02492357  0.9994417   0.        ] 
 [-0.09950941  0.00166177 -0.00128768  1.        ]]
'''

if __name__ == '__main__':
    sequence_index = 18
    toolset = 'gt'
    mode_name = 'stereo'
    set_save_pose = True
    set_enable_mapping = False
    set_enable_loop_closure = False    

    sequence, fx_val, fy_val, cx_val, cy_val, bf_val, resize_val, abs_resize_val = big_table[sequence_index]

    path_base = './data' #'E:/voldor_data/data'#'./data'
    pose_base = './poses'
    voc_file = 'hl2_5_ORBvoc.bin' #'./ORBvoc.bin'
    path_flow = os.path.join(path_base, sequence, f'flow_{toolset}')
    path_flow_2 = os.path.join(path_base, sequence, f'flow_2_{toolset}')
    path_disp = os.path.join(path_base, sequence, f'disp_{toolset}')
    #path_disp = os.path.join(path_base, sequence, f'disp')
    path_img = os.path.join(path_base, sequence, 'img')
    fname_pose = os.path.join(pose_base, f'pose_{sequence}_{mode_name}_{toolset}.txt')

    cmd = 'C:/Users/jdibe/AppData/Local/Programs/Python/Python36/python.exe D:/jcds/Documents/GitHub/xvoldor/demo/demo.py'
    fx = f'--fx {fx_val}'
    fy = f'--fy {fy_val}'
    cx = f'--cx {cx_val}'
    cy = f'--cy {cy_val}'
    bf = f'--bf {bf_val}'
    flow_dir = f'--flow_dir {path_flow}'
    flow_2_dir = f'--flow_2_dir {path_flow_2}'
    img_dir = f'--img_dir {path_img}'
    #disp_dir = f'--disp_dir {path_disp}'
    disp_dir = ''
    disp_dir = f'--disp_dir {path_disp}'
    mode = f'--mode {mode_name}'
    enable_mapping = '--enable_mapping' if (set_enable_mapping) else ''
    enable_loop_closure = f'--enable_loop_closure {voc_file}' if (set_enable_loop_closure) else ''
    resize = f'--resize {resize_val}'
    abs_resize = f'--abs_resize {abs_resize_val}'
    save_pose = f'--save_pose {fname_pose}' if (set_save_pose) else ''

    os.system(f'{cmd} {fx} {fy} {cx} {cy} {bf} {flow_dir} {flow_2_dir} {img_dir} {disp_dir} {mode} {enable_mapping} {enable_loop_closure} {resize} {abs_resize} {save_pose}')
