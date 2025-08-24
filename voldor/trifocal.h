
#pragma once

void compute_TFT(float const* points_2D, int count, float const* fx, float const* fy, float const* cx, float const* cy, float* const base_2D, float* const points_3D, float* out_TFT, float* RT01, float* RT02);
void print_TFT(float const* TFT);

