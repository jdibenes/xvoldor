
#pragma once

void compute_TFT(float const* points_2D, int count, float const* fx, float const* fy, float const* cx, float const* cy, float* const depth, float* out_TFT, float* RT01, float* RT02);
void print_TFT(float const* TFT);

