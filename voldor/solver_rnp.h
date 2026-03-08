
#pragma once

bool solver_r6p1l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12);
bool solver_r6p2l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12);
bool solver_r6p2i(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, int max_iterations);
