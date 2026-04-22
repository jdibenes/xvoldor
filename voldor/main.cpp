#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "voldor_utils.h"
#include "config.h"
#include "geometry.h"
#include "voldor.h"
#include "../gpu-kernels/gpu_kernels.h"
#include "py_export.h"
#include "lock.h"

using namespace cv;
using namespace std;

void load_file(char const* filename, void* buffer, int offset, int count);

Eigen::Matrix<float, 4, 4> load_pose(char const* filename)
{
	FILE* f = fopen(filename, "rb");
	if (f == NULL) { throw std::runtime_error(""); }
	Cleaner file_close([=]() { fclose(f); });

	Eigen::Matrix<float, 4, 4> pose;
	uint32_t total = 4 * 4;
	uint32_t count = fread(pose.data(), sizeof(float), total, f);
	if (count != total) { throw std::runtime_error(""); }

	pose.transposeInPlace();

	return pose;
}


int main(int argc, char* argv[]) {
	cout << "TODO: VOLDOR debug exec." << endl;

	//VOLDOR voldor(cfg);
	//voldor.init(flows, disparity, Mat(), depth_priors, depth_prior_poses, vector<Mat>());
	//voldor.solve();

	//voldor.save_result(output_dir);

	//
	char const* cfg = 
		"--silent --meanshift_kernel_var 0.1 --disp_delta 1 --delta 0.2 --max_iters 6 "
		"--pose_sample_min_depth 0.586270751953125 --pose_sample_max_depth 117.254150390625 ";

	char const* const path_flow = "";
	char const* const path_disp = "";
	float fx = 586.27075;
	float fy = 586.27075;
	float cx = 374.04108;//760 / 2;//374.04108;
	float cy = 202.26265;//428 / 2;//202.26265;
	float basefocal = 117.254150390625;
	int N = 5;
	int w = 760;
	int h = 428;

	int n_registered = -1;
	float* flows_pt = new float[N * w * h * 2];
	float* flows_2_pt = new float[N * w * h * 2];
	float* disparity_pt = new float[w * h * 2];
	float* disparities_pt = new float[(N + 1) * w * h * 2];

	float* poses = new float[N * 6];
	float* poses_covar = new float[N * 6 * 6];
	float* depth = new float[w * h];
	float* depth_conf = new float[w * h];
	memset(poses, 0, N * 6);

	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt/000061.flo", flows_pt + 0 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt/000062.flo", flows_pt + 1 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt/000063.flo", flows_pt + 2 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt/000064.flo", flows_pt + 3 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt/000065.flo", flows_pt + 4 * (w * h * 2), 12, -1);

	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt/000061.flo", flows_2_pt + 0 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt/000062.flo", flows_2_pt + 1 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt/000063.flo", flows_2_pt + 2 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt/000064.flo", flows_2_pt + 3 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt/000065.flo", flows_2_pt + 4 * (w * h * 2), 12, -1);

	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000061.flo", disparity_pt, 12, -1);

	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000061.flo", disparities_pt + 0 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000062.flo", disparities_pt + 1 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000063.flo", disparities_pt + 2 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000064.flo", disparities_pt + 3 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000065.flo", disparities_pt + 4 * (w * h * 2), 12, -1);
	load_file("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt/000066.flo", disparities_pt + 5 * (w * h * 2), 12, -1);

	// hl2_to_opencv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)
	Eigen::Matrix<float, 4, 4> hl2_to_opencv{
		{1,  0,  0,  0},
		{0, -1,  0,  0},
		{0,  0, -1,  0},
		{0,  0,  0,  1}
	};


	Eigen::Matrix<float, 4, 4> a_pose_1 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000061.bin").transpose() * hl2_to_opencv;
	Eigen::Matrix<float, 4, 4> a_pose_2 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000062.bin").transpose() * hl2_to_opencv;
	Eigen::Matrix<float, 4, 4> a_pose_3 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000063.bin").transpose() * hl2_to_opencv;
	Eigen::Matrix<float, 4, 4> a_pose_4 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000064.bin").transpose() * hl2_to_opencv;
	Eigen::Matrix<float, 4, 4> a_pose_5 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000065.bin").transpose() * hl2_to_opencv;
	Eigen::Matrix<float, 4, 4> a_pose_6 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000066.bin").transpose() * hl2_to_opencv;

	Eigen::Matrix<float, 4, 4> r_pose[] = {
		a_pose_2.inverse() * a_pose_1,
		a_pose_3.inverse() * a_pose_2,
		a_pose_4.inverse() * a_pose_3,
		a_pose_5.inverse() * a_pose_4,
		a_pose_6.inverse() * a_pose_5
	};

	for (int i = 0; i < (w * h); ++i)
	{
		disparity_pt[i] = -disparity_pt[2 * i];
	}

	for (int i = 0; i < ((N + 1) * w * h); ++i)
	{
		disparities_pt[i] = -disparities_pt[2 * i];
	}






	py_voldor_wrapper(
		flows_pt,
		flows_2_pt,
		disparities_pt,
		disparity_pt,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		fx, fy, cx, cy, basefocal, N, 0, w, h, cfg,
		n_registered,
		poses,
		poses_covar,
		depth,
		depth_conf
	);

	std::cout << "registered " << n_registered << std::endl;
	for (int i = 0; i < n_registered; ++i)
	{
		std::cout << "pose (" << i << ")" << std::endl;
		for (int j = 0; j < 6; ++j) {
			std::cout << poses[6 * i + j];
			if (j < 5) { std::cout << ", "; }
		}
		Eigen::Matrix<float, 3, 3> R_gt = r_pose[i](Eigen::seqN(0, 3), Eigen::seqN(0, 3));
		Eigen::Matrix<float, 3, 1> r_gt = vector_r_rodrigues(R_gt);
		Eigen::Matrix<float, 3, 1> t_gt = r_pose[i](Eigen::seqN(0, 3), 3);
		//std::cout << " | r_gt: " << r_gt << " | t_gt: " << t_gt;
		Eigen::Matrix<float, 2, 1> errors = compute_error(r_gt, t_gt, matrix_from_buffer<float, 3, 1>(poses + 6 * i), matrix_from_buffer<float, 3, 1>(poses + 6 * i + 3));
		std::cout << " | Errors: " << errors(0) << ", " << errors(1);
		std::cout << std::endl;
	}


	//char c;
	//std::cin >> c;

	return 0;
}
