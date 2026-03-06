
#pragma once

// OK
// p3d_1: 3xN
// p2d_2: 2xN
// p2d_3: 2xN
// r_12:  3x1
// t_12:  3x1
// r_23:  3x1
// t_23:  3x1
bool trifocal_R_t_linear(float const* p3d_1, float const* p2d_2, float const* p2d_3, int N, float* r_12, float* t_12, float* r_23, float* t_23, float threshold = 0);
