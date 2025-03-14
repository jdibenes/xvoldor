
import os
import shutil
import xv_file

path = './data/kitti-flow/training'
sequence = 'colored_0'

files = sorted(xv_file.scan_files(os.path.join(path, sequence)))

files_A = files[0::2]
files_B = files[1::2]

for file_A, file_B in zip(files_A, files_B):
    _, nameA, extA = xv_file.get_file_name(file_A)
    _, nameB, extB = xv_file.get_file_name(file_B)

    prefixA = nameA[:6]
    prefixB = nameB[:6]

    if (prefixA != prefixB):
        print(f'prefix mismatch {prefixA} : {prefixB}')
        break

    dst = os.path.join(path, sequence, prefixA)
    dstA = os.path.join(dst, f'{nameA}{extA}')
    dstB = os.path.join(dst, f'{nameB}{extB}')

    os.makedirs(dst, exist_ok=True)

    shutil.move(file_A, dstA)
    shutil.move(file_B, dstB)
