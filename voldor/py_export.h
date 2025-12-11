
#pragma once

extern
int
py_voldor_wrapper
(
	// inputs
	float const* flows_1_pt,
	float const* flows_2_pt,
	float const* disparities_pt,
	float const* disparity,
	float const* disparity_pconf,
	float const* depth_priors,
	float const* depth_prior_poses,
	float const* depth_prior_pconfs,
	float const fx,
	float const fy,
	float const cx,
	float const cy,
	float const basefocal,
	int const N,
	int const N_dp,
	int const w,
	int const h,
	char const* config,
	// outputs
	int& n_registered,
	float* poses,
	float* poses_covar,
	float* depth,
	float* depth_conf
);
