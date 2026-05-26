
import argparse
parser = argparse.ArgumentParser(description='VOLDOR-SLAM demo script')
parser.add_argument('--mode', type=str, required=True, help='One from stereo/mono-scaled/mono. For stereo and mono-scaled, disparity input will be required.')
parser.add_argument('--flow_dir', type=str, required=True)
parser.add_argument('--flow_2_dir', type=str, required=True)
parser.add_argument('--disp_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('--fx', type=float, required=True)
parser.add_argument('--fy', type=float, required=True)
parser.add_argument('--cx', type=float, required=True)
parser.add_argument('--cy', type=float, required=True)
parser.add_argument('--bf', type=float, default=0, help='Baseline x focal, which determines the world scale. If set to 0, default baseline is 0.')
parser.add_argument('--resize', type=float, default=0.5, help='resize input size')
parser.add_argument('--abs_resize', type=float, help='Resize factor related to the size that optical flow is estimated from. (useful to residual model)')
parser.add_argument('--enable_loop_closure', type=str, default=None)
parser.add_argument('--enable_mapping', action='store_true')
parser.add_argument('--save_poses', type=str)
parser.add_argument('--save_depths', type=str)
parser.add_argument('--multiview_mode', type=int)
parser.add_argument('--solver_select', type=int)
parser.add_argument('--batch_workers', type=int)
parser.add_argument('--batch_unique', action='store_true') #
parser.add_argument('--disparities_enable', action='store_true')
parser.add_argument('--disparities_use_0', action='store_true')
parser.add_argument('--rs_direction', type=int)
parser.add_argument('--rs_r0', type=float)
parser.add_argument('--rs_iterations', type=int)
parser.add_argument('--tf_threshold', type=float)
parser.add_argument('--tf_enable_next_pool', action='store_true')
parser.add_argument('--tf_enable_flow_2', action='store_true')
parser.add_argument('--tf_use_flow_2', action='store_true')
parser.add_argument('--tf_squared_error_threshold', type=float)
parser.add_argument('--tf_sample_size', type=int)
parser.add_argument('--estimate_intrinsics', action='store_true')
parser.add_argument('--square_pixels', action='store_true')
parser.add_argument('--shared_focals', action='store_true')
parser.add_argument('--depth_scale', type=float, default=1000)

opt = parser.parse_args()
if opt.abs_resize is None:
    opt.abs_resize = opt.resize

import sys
sys.path.append('../slam_py')
#sys.path.append('./lib_VS_flow2_fix_triangulation_mt_ressl')
#sys.path.append('./lib_p3p_but_not_planar')
#sys.path.append('./lib_gpm_3d3d')
#sys.path.append('./lib_vs_p3p_migration')
#sys.path.append('./lib_p3plt_gpu')
#sys.path.append('./lib_rnp')
#sys.path.append('./lib_build_260414')
#sys.path.append('./lib_poselib_test')
sys.path.append('./lib_focal_test_2026_05_26')
from voldor_viewer import VOLDOR_Viewer
from voldor_slam import VOLDOR_SLAM

import os
import time
import threading

if __name__ == '__main__':
    print(opt)

    # init slam instance and select mode from mono/mono-scaled/stereo
    slam = VOLDOR_SLAM(mode=opt.mode)

    extra_args = ''
    extra_args += f'--multiview_mode {opt.multiview_mode} --solver_select {opt.solver_select} '
    extra_args += f'--batch_workers {opt.batch_workers} ' + ('--batch_unique' if (opt.batch_unique) else '') + f' '
    extra_args += ('--disparities_enable' if (opt.disparities_enable) else '') + f' ' + ('--disparities_use_0' if (opt.disparities_use_0) else '') + f' '
    extra_args += f'--rs_direction {opt.rs_direction} --rs_r0 {opt.rs_r0} --rs_iterations {opt.rs_iterations} '
    extra_args += f'--tf_threshold {opt.tf_threshold} ' + ('--tf_enable_next_pool' if (opt.tf_enable_next_pool) else '') + f' ' + ('--tf_enable_flow_2' if (opt.tf_enable_flow_2) else '') + f' ' + ('--tf_use_flow_2' if (opt.tf_use_flow_2) else '') + f' --tf_squared_error_threshold {opt.tf_squared_error_threshold} --tf_sample_size {opt.tf_sample_size} '
    extra_args += ('--estimate_intrinsics' if (opt.estimate_intrinsics) else '') + f' ' + ('--square_pixels' if (opt.square_pixels) else '') + f' ' + ('--shared_focals' if (opt.shared_focals) else '') + f' '

    # set camera intrinsic
    slam.set_cam_params(opt.fx,opt.fy,opt.cx,opt.cy,opt.bf, rescale=opt.resize, depth_scale=opt.depth_scale)
    slam.voldor_user_config = f'--abs_resize_factor {opt.abs_resize} ' + extra_args
    print(slam.voldor_user_config)

    # enable loop closure
    if opt.enable_loop_closure is not None:
        slam.enable_loop_closure(opt.enable_loop_closure)

    # start flow loader
    threading.Thread(target=slam.flow_loader, kwargs={'flow_path':opt.flow_dir, 'resize':opt.resize}).start()
    slam.flow_loader_sync(0, block_when_uninit=True)

    # start image loader
    if opt.img_dir is not None:
        threading.Thread(target=slam.image_loader, kwargs={'image_path':opt.img_dir}).start()
        slam.image_loader_sync(0, block_when_uninit=True)
        slam.use_image_info=True
    else:
        slam.use_image_info=False
    
    # start disparity loader
    if opt.disp_dir is not None:
        threading.Thread(target=slam.disp_loader, kwargs={'disp_path':opt.disp_dir}).start()    
        slam.disp_loader_sync(0, block_when_uninit=True)

    # start flow 2 loader
    threading.Thread(target=slam.flow_2_loader, kwargs={'flow_2_path':opt.flow_2_dir, 'resize':opt.resize}).start()
    slam.flow_2_loader_sync(0, block_when_uninit=True)
    
    # start viewer
    viewer = VOLDOR_Viewer(slam)
    viewer_thread = threading.Thread(target=viewer.start)
    viewer_thread.start()
    
    # start VO and mapping threads
    vo_thread = threading.Thread(target=slam.vo_thread)
    vo_thread.start()
    if opt.enable_mapping:
        mapping_thread = threading.Thread(target=slam.mapping_thread)
        mapping_thread.start()

    # wait them to end
    vo_thread.join()
    if opt.enable_mapping:
        mapping_thread.join()

    # save poses and depths
    if opt.save_poses is not None:
        slam.save_poses(opt.save_poses, format='KITTI')
    if opt.save_depths is not None:
        slam.save_depth_maps(opt.save_depths)
