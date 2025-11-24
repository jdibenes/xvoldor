#include "py_export.h"
#include "voldor.h"
#include <iterator>

int
py_voldor_wrapper
(
	// inputs
	const float* flows_pt,
	const float* disparity_pt,
	const float* disparity_pconf_pt,
	const float* depth_priors_pt,
	const float* depth_prior_poses_pt,
	const float* depth_prior_pconfs_pt,
	const float fx,
	const float fy,
	const float cx,
	const float cy,
	const float basefocal,
	const int N,
	const int N_dp,
	const int w,
	const int h,
	const char* config_pt,
	// outputs
	int& n_registered,
	float* poses_pt,
	float* poses_covar_pt,
	float* depth_pt,
	float* depth_conf_pt,
	// Extended
	// inputs
	const float* flows_2_pt,
	const float* disparities_pt
)
{

	Config cfg;
	std::istringstream iss(config_pt);
	std::vector<std::string> cfg_strs(
		std::istream_iterator<std::string>{iss},
		std::istream_iterator<std::string>());
	cfg.fx = fx;
	cfg.cx = cx;
	cfg.fy = fy;
	cfg.cy = cy;
	cfg.basefocal = basefocal;
	cfg.read_config(cfg_strs);


	std::vector<cv::Mat> flows;
	cv::Mat disparity;
	cv::Mat disparity_pconf;
	std::vector<cv::Mat> depth_priors;
	std::vector<cv::Vec6f> depth_prior_poses;
	std::vector<cv::Mat> depth_priors_pconfs;
	std::vector<cv::Mat> flows_2;
	std::vector<cv::Mat> disparities;

	for (int i = 0; i < N; i++) {
		flows.push_back(cv::Mat(cv::Size(w, h), CV_32FC2, (void*)(flows_pt + i * w * h * 2)));
		flows_2.push_back(cv::Mat(cv::Size(w, h), CV_32FC2, (void*)(flows_2_pt + i * w * h * 2)));
	}

	if (disparity_pt)
		disparity = cv::Mat(cv::Size(w, h), CV_32F, (void*)disparity_pt);
	if (disparity_pconf_pt)
		disparity_pconf = cv::Mat(cv::Size(w, h), CV_32F, (void*)disparity_pconf_pt);

	for (int i = 0; i < N_dp; i++) {
		depth_priors.push_back(cv::Mat(cv::Size(w, h), CV_32F, (void*)(depth_priors_pt + i * w*h)));
		depth_prior_poses.push_back(cv::Vec6f(depth_prior_poses_pt + i * 6));
		if (depth_prior_pconfs_pt)
			depth_priors_pconfs.push_back(cv::Mat(cv::Size(w, h), CV_32F, (void*)(depth_prior_pconfs_pt + i * w*h)));
	}

	if (disparities_pt)
	{
		for (int i = 0; i < N; ++i)
		{
			disparities.push_back(cv::Mat(cv::Size(w, h), CV_32F, (void*)(disparities_pt + i * w * h)));
		}
	}


	VOLDOR voldor(cfg);
	voldor.init(flows, disparity, disparity_pconf, depth_priors, depth_prior_poses, depth_priors_pconfs, flows_2, disparities);
	voldor.solve();

	n_registered = voldor.n_flows;

	for (int i = 0; i < voldor.n_flows; i++) {
		if (poses_pt)
			memcpy(poses_pt + i * 6, voldor.cams[i].pose6().val, 6 * sizeof(float));
		if (poses_covar_pt)
			memcpy(poses_covar_pt + i * 6 * 6, voldor.cams[i].pose_covar.data, 6 * 6 * sizeof(float));
	}

	if (depth_pt)
		memcpy(depth_pt, voldor.depth.data, w*h * sizeof(float));

	if (depth_conf_pt) {
		cv::Mat depth_conf = cv::Mat::zeros(cv::Size(w, h), CV_32F);
		for (int i = 0; i < voldor.n_flows; i++)
			depth_conf += voldor.rigidnesses[i];
		for (int i = 0; i < voldor.n_depth_priors; i++)
			depth_conf += voldor.depth_prior_confs[i];
		depth_conf /= (float)(voldor.n_flows + voldor.n_depth_priors);
		memcpy(depth_conf_pt, depth_conf.data, w*h * sizeof(float));
	}

	return 0;
}