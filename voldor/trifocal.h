
#pragma once

bool
trifocal_R_t_linear
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p3d_1,
    int N,
    bool use_prior,
    float* r1,
    float* t1,
    float* r2,
    float* t2
);

bool trifocal_R_t(float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* sp2d, float const* sp3d, float* tft, float* rt1, float* rt2, float* s1, float* s2);
int trifocal_R_t_batch(int jobs, int workers, float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* p2d_s, float const* p3d_s, int count, float* tft, float* rt1, float* rt2, float* s1, float* s2);
