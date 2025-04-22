
import numpy as np
import os
import cv2
import xv_file
import xv_flow

toolset = 'ptl-pwcnet'
delay = 0
path_sequence = '../demo/data/hl2_4'
max_error = 5

#------------------------------------------------------------------------------

path_img = os.path.join(path_sequence, 'img')
path_flow1 = os.path.join(path_sequence, f'flow_{toolset}')
path_flow2 = os.path.join(path_sequence, f'flow_2_{toolset}')

files_img = sorted(xv_file.scan_files(path_img))
files_flow1 = sorted(xv_file.scan_files(path_flow1))
files_flow2 = sorted(xv_file.scan_files(path_flow2))

files = zip(files_img[:-2], files_img[1:-1], files_img[2:], files_flow1[:-1], files_flow1[1:], files_flow2)

LD = True

for file_img0, file_img1, file_img2, file_flow10, file_flow11, file_flow20 in files:
    flow10 = xv_flow.flo_to_flow(file_flow10)
    flow11 = xv_flow.flo_to_flow(file_flow11)
    flow20 = xv_flow.flo_to_flow(file_flow20)

    if (LD):
        x = np.arange(flow10.shape[1], dtype=np.float32)
        y = np.arange(flow10.shape[0], dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        xy = np.dstack((xx, yy))
        shape = (flow10.shape[1], flow10.shape[0])

    img0 = cv2.resize(cv2.imread(file_img0), shape, interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(cv2.imread(file_img1), shape, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(cv2.imread(file_img2), shape, interpolation=cv2.INTER_LINEAR)

    map10 = xy + flow10
    map11 = xy + flow10 + flow11
    map20 = xy + flow20

    error20 = np.linalg.norm(flow10 + flow11 - flow20, axis=-1)
    
    error20[error20 > max_error] = max_error
    error20 = ((error20 / max_error) * 255).astype(np.uint8)

    cv2.imshow('img0', img0)
    cv2.imshow('flow10_w', cv2.remap(img1, map10[:, :, 0], map10[:, :, 1], cv2.INTER_LINEAR))
    cv2.imshow('flow11_w', cv2.remap(img2, map11[:, :, 0], map11[:, :, 1], cv2.INTER_LINEAR))
    cv2.imshow('flow20_w', cv2.remap(img2, map20[:, :, 0], map20[:, :, 1], cv2.INTER_LINEAR))
    cv2.imshow('error20', cv2.applyColorMap(error20, cv2.COLORMAP_INFERNO))
    
    if (cv2.waitKey(delay) == 27):
        break
