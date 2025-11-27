#pragma once
#include "utils.h"
#include "config.h"

int 
optimize_camera_pose
(
	std::vector<cv::Mat> const& flows,
	std::vector<cv::Mat> const& rigidnesses, 
	cv::Mat const& depth,
	std::vector<Camera>& cams,
	int n_flows,
	int active_idx,
	bool successive_pose,
	bool rg_refine,
	bool update_batch_instance,
	bool update_iter_instance,
	Config const& cfg,
	std::vector<cv::Mat> const& flows_2,
	std::vector<cv::Mat> const& disparities
);

void estimate_depth_closed_form(cv::Mat flow, cv::Mat& depth, Camera cam,
	float min_depth = 1e-2f, float max_depth = 1e10f);


void estimate_camera_pose_epipolar(cv::Mat flow, Camera& cam,
	cv::Mat mask = cv::Mat(), int sampling_2d_step = 4);
