#pragma once
#include "voldor_utils.h"
#include "config.h"

int
optimize_camera_pose
(
	std::vector<cv::Mat> const& flows_1,
	std::vector<cv::Mat> const& flows_2,
	std::vector<cv::Mat> const& disparities,
	std::vector<cv::Mat> const& rigidnesses,
	cv::Mat const& depth,
	std::vector<Camera>& cams, // MODIFIED
	int n_flows,
	int active_idx,
	bool successive_pose,
	bool rg_refine,
	bool update_batch_instance, // !cfg.exclusive_gpu_context || (iters_cur == 1 && i == 0)
	bool update_iter_instance, // i == 0 : true for first flow
	Config const& cfg,
	cv::Mat& next_pool,
	int& next_pool_used
);

void estimate_depth_closed_form(cv::Mat flow, cv::Mat& depth, Camera cam,
	float min_depth = 1e-2f, float max_depth = 1e10f);


void estimate_camera_pose_epipolar(cv::Mat flow, Camera& cam,
	cv::Mat mask = cv::Mat(), int sampling_2d_step = 4);
