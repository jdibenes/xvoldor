#pragma once
#include "utils.h"
#include "config.h"
#include "geometry.h"
#include "kitti.h"
#include "../gpu-kernels/gpu_kernels.h"

enum OPTIMIZE_DEPTH_FLAG {
	OD_DEFAULT = 0,
	OD_ONLY_USE_DEPTH_PRIOR = 1,
	OD_UPDATE_RIGIDNESS_ONLY = 2,
};


class VOLDOR {
public:

	int n_flows, n_flows_init;
	int n_depth_priors;
	int w, h;
	int iters_cur;
	int iters_remain;
	Config cfg;
	bool has_disparity;

	
	std::vector<cv::Mat> flows_1;
	std::vector<cv::Mat> flows_2;
	std::vector<cv::Mat> disparities;
	std::vector<cv::Mat> depth_priors;
	std::vector<Camera> depth_prior_poses;
	std::vector<cv::Mat> depth_prior_pconfs;
	std::vector<cv::Mat> depth_prior_confs;

	cv::Mat depth;
	std::vector<cv::Mat> rigidnesses;
	std::vector<Camera> cams;
	

	KittiGround ground;

	VOLDOR(Config cfg, bool exclusive_gpu_context = true) :
		cfg(cfg) {
		if (!cfg.silent)
			cfg.print_info();
	}

	void
	init
	(
		std::vector<cv::Mat> _flows_1,
		std::vector<cv::Mat> _flows_2 = std::vector<cv::Mat>(),
		std::vector<cv::Mat> _disparities = std::vector<cv::Mat>(),
		cv::Mat _disparity = cv::Mat(),
		cv::Mat _disparity_pconf = cv::Mat(),
		std::vector<cv::Mat> _depth_priors = std::vector<cv::Mat>(),
		std::vector<cv::Vec6f> _depth_prior_poses = std::vector<cv::Vec6f>(),
		std::vector<cv::Mat> _depth_prior_pconfs = std::vector<cv::Mat>()
	);

	int solve();


	void bootstrap();

	void optimize_cameras();

	void optimize_depth(OPTIMIZE_DEPTH_FLAG flag=OD_DEFAULT);

	void normalize_world_scale();

	void estimate_kitti_ground();

	void save_result(std::string save_dir);

	void debug();


private:
#if defined(WIN32) || defined(_WIN32)
	std::chrono::time_point<std::chrono::steady_clock> time_stamp;
#else
	std::chrono::system_clock::time_point time_stamp;
#endif
	void tic() {
		time_stamp = std::chrono::high_resolution_clock::now();
	}

	float toc() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6f;
	}
	void toc(std::string job_name) {
		std::cout << job_name << " elapsed time = " << toc() << "ms." << std::endl;
	}

};
