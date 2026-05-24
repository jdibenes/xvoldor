
#pragma once

bool solver_gpm_hpc0(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc1(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc2(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc3(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);

bool solver_gpm_nm5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm6(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm7(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);

bool solver_rnp_r6p1l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12);
bool solver_rnp_r6p2l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12);
bool solver_rnp_r6p2i(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12, int max_iterations);

bool solver_gpm_m4(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_rpe_m5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);

bool solver_tft_linear(float const* p3d_1, float const* p2d_2, float const* p2d_3, int N, float* r_12, float* t_12, float* r_23, float* t_23, float threshold);
bool solver_tft_p4p(float const* p3d_1, float const* p2d_2, float const* p2d_3, float* r_12, float* t_12, float* r_23, float* t_23);
bool solver_tft_4p3vpara(float const* p3d_1, float const* p2d_2, float const* p2d_3, float* r_12, float* t_12, float* r_23, float* t_23, int iterations);

bool solver_p4p_lambdatwist(float const* p3d_1, float const* p2d_2, float* r_12, float* t_12);
bool solver_p4p_ap3p(float const* p3d_1, float const* p2d_2, float* r_12, float* t_12);

bool solver_ppf_p4pf(float const* p3d_1, float const* p2k_2, bool same, float cx, float cy, float* r_12, float* t_12, float* f_xy);

bool solver_rpf_r7pfi(float const* p3d_1, float const* p2k_2, float cx, float cy, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12, float* f_xy, int max_iterations);
