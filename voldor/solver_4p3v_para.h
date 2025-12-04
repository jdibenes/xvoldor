
#pragma once

bool
solver_4p3v_para
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p3d_1,
    bool use_prior,
    int N,
    float* r1,
    float* t1,
    float* r2,
    float* t2
);