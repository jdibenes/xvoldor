
import cv2
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import xv_file
import xv_flow
import flow_viz
import os

def build_flow(model_name, ckpt_path, sequence, set_disp, factor):
    path_in_data = 'C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/kitti-flow/training/colored_0'
    path_out_data = 'C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/kitti-flow/training'
    path_1 = os.path.join(path_in_data, f'{sequence}')
    flow_name = 'flow'
    path_out = os.path.join(path_out_data, f'{flow_name}_ptl-{model_name}', )
    path_out_vis = os.path.join(path_out_data, f'{flow_name}_vis_ptl-{model_name}')

    images_1 = sorted(xv_file.scan_files(path_1))
    images_2 = images_1[1:]
    images_1 = images_1[:-1]

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
    param_model_name = 'maskflownet' #'pwcnet' #'neuflow2' #'memflow' #'pwcnet' #'memflow'
    param_ckpt_path = 'kitti' #'sintel' #'mixed'#'spring' #'sintel' #'spring'
    param_factor = 1

    #sequences = ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4', 'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']
    sequences = [f'{sequence_number:06}' for sequence_number in range(0, 194)]

    for n_sequence in sequences:
        print(f'generating {n_sequence}')
        build_flow(param_model_name, param_ckpt_path, n_sequence, False, param_factor)










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