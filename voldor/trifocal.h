
#pragma once

void
trifocal_R_t
(
    float const* points_2D,
    int count,
    float const* fx,
    float const* fy,
    float const* cx,
    float const* cy,
    float* const map_2D,
    float* const map_3D,
    float* out_TFT,
    float* out_r01,
    float* out_t01,
    float* out_r02,
    float* out_t02
);

void print_TFT(float const* TFT);
