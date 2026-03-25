from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy

import shutil

# Change these folders
opencv_include_dir = 'C:/Users/jcds/SDK/opencv_3_4_16/build/include'
ceres_include_dirs = [
    'C:/Users/jcds/Documents/GitHub/vcpkg/packages/ceres_x64-windows/include',
    'C:/Users/jcds/Documents/GitHub/vcpkg/packages/eigen3_x64-windows/include/eigen3',
    'C:/Users/jcds/Documents/GitHub/vcpkg/packages/glog_x64-windows/include'
]

opencv_lib_dir = 'C:/Users/jcds/SDK/opencv_3_4_16/build/x64/vc15/lib'
ceres_lib_dirs = ['C:/Users/jcds/Documents/GitHub/vcpkg/packages/ceres_x64-windows/lib', 'C:/Users/jcds/Documents/GitHub/vcpkg/packages/glog_x64-windows/lib']

opencv_lib_name = 'opencv_world3416'
ceres_lib_names = ['ceres', 'glog']

# Change this if you target different host/device
nvcc_machine_code = '' #'-m64 -arch=compute_61 -code=sm_61'

gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

gpu_kernel_build_cmd = f'nvcc {gpu_sources_cpp} {gpu_sources_cu} -I {opencv_include_dir} -L {opencv_lib_dir} -l {opencv_lib_name} \
                        -shared -o ./gpu-kernels.dll -O3 -cudart static {nvcc_machine_code} -Xcompiler /wd4819'
os.system(gpu_kernel_build_cmd)

ext = Extension('pyvoldor_full',
    sources = ['pyvoldor_full.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../frame-alignment/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../pose-graph/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../thirdparty/rnp/*.cpp')],
    language = 'c++',
    library_dirs = ['./gpu-kernels.lib', opencv_lib_dir] + ceres_lib_dirs,
    libraries = ['gpu-kernels', opencv_lib_name] + ceres_lib_names,
    include_dirs = [numpy.get_include(), opencv_include_dir] + ceres_include_dirs + ['../../thirdparty'],
    extra_compile_args = ['/std:c++17', '-MD'],
    define_macros = [('_CRT_NONSTDC_NO_DEPRECATE',''), ('CERES_USE_CXX_THREADS',''), ('GLOG_USE_GLOG_EXPORT','')]
)

setup(
    name='pyvoldor_full',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)

