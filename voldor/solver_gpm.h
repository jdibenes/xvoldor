
#pragma once

bool solver_gpm_hpc0(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc1(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);
bool solver_gpm_hpc2(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12);

bool solver_gpm_nm5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm6(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
bool solver_gpm_nm7(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12);
