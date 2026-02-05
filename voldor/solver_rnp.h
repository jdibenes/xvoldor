
#pragma once

bool solver_r6p1l(float const* p3d, float const* p2d, bool direction, float r0, int maxpow, float* r01, float* t01);
bool solver_r6p2l(float const* p3d, float const* p2d, bool direction, float r0, float* r01, float* t01);
bool solver_r6pi(float const* p3d, float const* p2d, bool direction, float r0, int max_iterations, float* r01, float* t01);
