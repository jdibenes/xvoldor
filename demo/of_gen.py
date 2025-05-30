
import cv2
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import xv_file
import xv_flow
import flow_viz
import os

def build_flow(model_name, ckpt_path, sequence, set_disp, factor, stride):
    path_data = './data'
    path_1 = os.path.join(path_data, f'{sequence}/img')
    path_2 = os.path.join(path_data, f'{sequence}/imgR')
    flow_name = 'disp' if (set_disp) else 'flow'
    path_out = os.path.join(path_data, f'{sequence}/{flow_name}_{stride}_ptl-{model_name}' if (stride > 1) else f'{sequence}/{flow_name}_ptl-{model_name}')
    path_out_vis = os.path.join(path_data, f'{sequence}/vis_{flow_name}_{stride}_ptl-{model_name}' if (stride > 1) else f'{sequence}/vis_{flow_name}_ptl-{model_name}')

    images_1 = sorted(xv_file.scan_files(path_1))
    if (set_disp):
        images_2 = sorted(xv_file.scan_files(path_2))
    else:
        images_2 = images_1[stride:]
        images_1 = images_1[:-stride]

    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out_vis, exist_ok=True)

    # Get an optical flow model. As as example, we will use RAFT Small
    # with the weights pretrained on the FlyingThings3D dataset
    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)

    for fname_1, fname_2 in zip(images_1, images_2):
        _, name_1, _ = xv_file.get_file_name(fname_1)
        _, name_2, _ = xv_file.get_file_name(fname_2)

        # Load the images
        images = [
            cv2.resize(cv2.imread(fname_1), dsize=(0,0), fx=factor, fy=factor),
            cv2.resize(cv2.imread(fname_2), dsize=(0,0), fx=factor, fy=factor),
        ]

        # A helper to manage inputs and outputs of the model
        io_adapter = IOAdapter(model, images[0].shape[:2])

        # inputs is a dict {'images': torch.Tensor}
        # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
        # (1, 2, 3, H, W)
        inputs = io_adapter.prepare_inputs(images)

        # Forward the inputs through the model
        predictions = model(inputs)

        # The output is a dict with possibly several keys,
        # but it should always store the optical flow prediction in a key called 'flows'.
        flows = predictions['flows']

        # flows will be a 5D tensor BNCHW.
        # This example should print a shape (1, 1, 2, H, W).
        #print(flows.shape)
        flow = flows[0, 0].permute(1, 2, 0).detach().cpu().numpy()

        xv_flow.flow_to_flo(flow, os.path.join(path_out, f'{name_1}.flo'))
        flow_vis = flow_viz.flow_to_image(flow, convert_to_bgr=True)
        cv2.imwrite(os.path.join(path_out_vis, f'{name_1}.jpg'), flow_vis)

        print(f'{name_1} -> {name_2}')


if __name__ == '__main__':
    #param_model_name = 'memflow'
    #param_ckpt_path = 'spring'
    #param_model_name = 'neuflow2'
    #param_ckpt_path = 'mixed'
    #param_model_name = 'pwcnet'
    #param_ckpt_path = 'sintel'
    param_model_name = 'maskflownet'
    param_ckpt_path = 'kitti'
    param_factor = 1.0
    stride = 1
    #param_factor = 1.0

    #sequences = ['KITTI-00','KITTI-01','KITTI-02','KITTI-04','KITTI-06','KITTI-07','KITTI-08','KITTI-09','KITTI-10']
    #sequences = ['KITTI-03', 'KITTI-05']
    #sequences = ['KITTI-00','KITTI-01','KITTI-02','KITTI-03','KITTI-04','KITTI-06','KITTI-07','KITTI-08','KITTI-09','KITTI-10']
    #sequences = ['c1', 'c2', 'c3', 'c4']
    #sequences = ['KITTI-03', 'KITTI-05']
    sequences = ['c1', 'hl2_4']
    sets_disp = [False]
    sets_stride = [1, 2]

    for stride in sets_stride:
        for n_sequence in sequences:
            for n_set_disp in sets_disp:
                print(f'generating {n_sequence} disp={n_set_disp}')
                build_flow(param_model_name, param_ckpt_path, n_sequence, n_set_disp, param_factor, stride)










#images_1 = [f'./data/{sequence}/img/000000.png']#
#images_2 = [f'./data/{sequence}/img/000001.png']#





#ptlflow.download_scripts()
#print(ptlflow.get_model_names())

'''
['ccmr', 'ccmr_p', 'craft', 'csflow', 'dicl', 'dip', 'fastflownet', 'flow1d', 'flowformer', 'flowformer_pp', 'flownet2', 'flownetc', 'flownetcs', 'flownetcss','lcv_raft', 'lcv_raft_small', 'liteflownet 'flownets', 'flownetsd', 'gma', 'gmflow', 'gmflow_p', 'gmflow_p_sc2', 'gmflow_p_sc2_ref6', 'gmflow_refine', 'gmflownet', 'gmflownet_mix', 'hd3', 'hd3_ctxt', 'maskflownet', 'maskflownet_s', 'matchflow'irr_pwc', 'irr_pwcnet', 'irr_pwcnet_irr', 'lcv_raft', 'lcv_raft_small', 'liteflownet', 'liteflownet2', 'liteflownet2_pseudoreg', 'liteflownet3', 'liteflownet3_apidflow_it2', 'rapidflow_it3', 'rapidflowpseudoreg', 'liteflownet3s', 'liteflownet3s_pseudoreg', 'llaflow', 'llaflow_raft', 'maskflownet', 'maskflownet_s', 'matchflow', 'matchflow_raft', 'memflow', 'mch', 'unimatch_sc2', 'unimatch_sc2_ref6', emflow_t', 'ms_raft_p', 'neuflow', 'neuflow2', 'pwcnet', 'pwcnet_nodc', 'raft', 'raft_small', 'rapidflow', 'rapidflow_it1', 'rapidflow_it2', 'rapidflow_it3', 'rapidflow_it6', 'rpknet', 'scopeflow', 'scv4', 'scv8', 'sea_raft', 'sea_raft_l', 'sea_raft_m', 'sea_raft_s', 'separableflow', 'skflow', 'splatflow', 'starflow', 'unimatch', 'unimatch_sc2', 'unimatch_sc2_ref6', 'vcn', 'vcn_small', 'videoflow_bof', 'videoflow_mof']
'''

# raft -> {chairs,things,sintel,kitti}


    #print(flow.shape)




    # Create an RGB representation of the flow to show it on the screen
    ##flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    ##flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    ##flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    ##flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

    # Show on the screen
    ##cv.imshow('image1', images[0])
    ##cv.imshow('image2', images[1])
    ##cv.imshow('flow', flow_bgr_npy)
    ##cv.waitKey()