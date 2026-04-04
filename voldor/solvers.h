
#pragma once

bool solver_gpm_hpc0(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc1(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc2(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);

bool solver_gpm_nm5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm6(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm7(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);

bool solver_r6p1l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12);
bool solver_r6p2l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12);
bool solver_r6p2i(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, int max_iterations);

bool solver_gpm_m4(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_rpe_m5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);

// OK
// p3d_1: 3xN
// p2d_2: 2xN
// p2d_3: 2xN
// r_12:  3x1
// t_12:  3x1
// r_23:  3x1
// t_23:  3x1
bool solver_tft_linear(float const* p3d_1, float const* p2d_2, float const* p2d_3, int N, float* r_12, float* t_12, float* r_23, float* t_23, float threshold = 0);
