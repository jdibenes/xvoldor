#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "config.h"
#include "geometry.h"
#include "voldor.h"
#include "py_export.h"
#include "helpers_lock.h"
#include "helpers_file.h"

using namespace cv;
using namespace std;



int main(int argc, char* argv[])
{
	char const* cfg =
		"--silent --meanshift_kernel_var 0.1 --disp_delta 1 --delta 0.2 --max_iters 5 "
		"--pose_sample_min_depth 0.586270751953125 --pose_sample_max_depth 117.254150390625 "
		"--multiview_mode 2 --solver_select 3 --batch_workers 18 --disparities_enable ";
		//"--multiview_mode 2 --solver_select 3 --batch_workers 18 ";
		//"--multiview_mode 3 --solver_select 26 --batch_workers 18 ";
	    
		//"--multiview_mode 3 --solver_select 24 --tf_sample_size 9 --batch_workers 18 --disparities_enable --tf_enable_flow_2 "; // --tf_enable_next_pool --tf_use_flow_2 

	int N = 5;
	int w = 760;
	int h = 428;

	float fx = 586.27075f;
	float fy = 586.27075f;
	float cx = 374.04108f;
	float cy = 202.26265f;
	float basefocal = 117.254150390625;
	
	//float cx = w / 2.0;
	//float cy = h / 2.0;
	//float basefocal = 0.2000000006662877f; // for estimate intrinsics
	
	int fid = 61;
	int last = 250;

	char const* const flow_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_gt";
	char const* const flow_2_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/flow_2_gt";
	char const* const disp_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/disp_gt";
	char const* const poses_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose";
	char const* const pattern = "%06d.flo";
	char const* const pattern_poses = "%06d.bin";
	
	std::unique_ptr<float[]> flows_1_pt = std::make_unique<float[]>(N * w * h * 2);
	std::unique_ptr<float[]> flows_2_pt = std::make_unique<float[]>((N - 1) * w * h * 2);
	std::unique_ptr<float[]> disparity_pt = std::make_unique<float[]>(w * h * 2);
	std::unique_ptr<float[]> disparities_pt = std::make_unique<float[]>((N + 1) * w * h * 2);
	std::unique_ptr<float[]> poses_pt = std::make_unique<float[]>((N + 1) * 16);

	std::unique_ptr<float[]> poses = std::make_unique<float[]>(N * 6);
	std::unique_ptr<float[]> poses_covar = std::make_unique<float[]>(N * 6 * 6);
	std::unique_ptr<float[]> depth = std::make_unique<float[]>(w * h);
	std::unique_ptr<float[]> depth_conf = std::make_unique<float[]>(w * h);

	float focals[2];

	float mean_r_error = 0;
	float mean_t_error = 0;
	float mean_a_error = 0;
	float mean_f_error = 0;
	float frame_count = 0;
	float fail_count = 0;

	//float max_t_ang_error = 0;

	while ((fid + N) < last)
	{
		auto time_stamp = std::chrono::high_resolution_clock::now();

		int n_registered = -1;

		load_window_flow(flow_path, pattern, fid, fid + N, flows_1_pt.get(), false, w, h);		
		load_window_flow(flow_2_path, pattern, fid, fid + N - 1, flows_2_pt.get(), false, w, h);		
		load_window_flow(disp_path, pattern, fid, fid + 1, disparity_pt.get(), false, w, h);		
		load_window_flow(disp_path, pattern, fid, fid + N + 1, disparities_pt.get(), false, w, h);		
		load_window_pose_hl2ss(poses_path, pattern_poses, fid, fid + N + 1, poses_pt.get());

		for (int i = 0; i <           (w * h); ++i) { disparity_pt[i]   = -disparity_pt[2 * i]; }
		for (int i = 0; i < ((N + 1) * w * h); ++i) { disparities_pt[i] = -disparities_pt[2 * i]; }

		py_voldor_wrapper(
			flows_1_pt.get(),
			flows_2_pt.get(),
			disparities_pt.get(),
			disparity_pt.get(),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			fx, fy, cx, cy, basefocal, N, 0, w, h, cfg,
			n_registered,
			poses.get(),
			poses_covar.get(),
			depth.get(),
			depth_conf.get(),
			focals
		);

		std::cout << "fid " << fid << " registered " << n_registered << std::endl;
		std::cout << "focals " << focals[0] << ", " << focals[1] << std::endl;
		if (n_registered <= 0) {
			fid++;
			fail_count++;
			continue;
			//break;
		}
		fid += n_registered;
		frame_count += n_registered;
		for (int i = 0; i < n_registered; ++i)
		{
			std::cout << "pose (" << i << ")" << std::endl;
			for (int j = 0; j < 6; ++j) {
				std::cout << poses[6 * i + j];
				if (j < 5) { std::cout << ", "; }
			}
			//*
			Eigen::Matrix<float, 4, 4> gt_relpose = matrix_from_buffer<float, 4, 4>(poses_pt.get() + (i * 16));
			Eigen::Matrix<float, 3, 3> R_gt = gt_relpose(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
			Eigen::Matrix<float, 3, 1> r_gt = vector_r_rodrigues(R_gt);
			Eigen::Matrix<float, 3, 1> t_gt = gt_relpose(Eigen::seqN(0, 3), 3);
			Eigen::Matrix<float, 3, 1> r_et = matrix_from_buffer<float, 3, 1>(poses.get() + 6 * i);
			Eigen::Matrix<float, 3, 1> t_et = matrix_from_buffer<float, 3, 1>(poses.get() + 6 * i + 3);
			//std::cout << " | r_gt: " << r_gt << " | t_gt: " << t_gt;
			Eigen::Matrix<float, 2, 1> errors = compute_error(r_gt, t_gt, r_et, t_et);
			float rad_to_deg = (180.0f / 3.14159265359f);
			float ang_error = std::acos(clamp(t_gt.normalized().dot(t_et.normalized()),-1.0f, 1.0f)) * rad_to_deg;
			std::cout << " | Errors: " << (errors(0) * rad_to_deg) << ", " << errors(1) << " (" << ang_error << ")";

			mean_r_error += (errors(0) * rad_to_deg);
			mean_t_error += errors(1);
			mean_a_error += ang_error;
			mean_f_error += abs(focals[0] - fx);

			std::cout << std::endl;
			//if (ang_error > max_t_ang_error) { max_t_ang_error = ang_error; }
			//*/
		}

		std::cout << "batch time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
	}

	std::cout << "mean_r_error: " << (mean_r_error / frame_count) << std::endl;
	std::cout << "mean_t_error: " << (mean_t_error / frame_count) << std::endl;
	std::cout << "mean_a_error: " << (mean_a_error / frame_count) << std::endl;
	std::cout << "mean_f_error: " << (mean_f_error / frame_count) << std::endl;
	std::cout << "frame_count: " << frame_count << " - " << " fail count: " << fail_count << std::endl;

	//std::cout << "MAX t ANG ERROR " << max_t_ang_error << std::endl;

	return 0;
}





//char path[260];

//370.00048828125, 370.00048828125, 477.6654968261719, 270.8048095703125, 44.458960801608859567705078125
	//float fx = 370.00048828125;
	//float fy = 370.00048828125;
	//float cx = 477.6654968261719;//760.0 / 2.0;//374.04108;
	//float cy = 270.8048095703125;//428.0 / 2.0;//202.26265;
	//float basefocal = 44.458960801608859567705078125;
	//int w = 960;
	//int h = 540;

	//int fid = 0;
	//int last = 4000;

	//char const* const flow_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/zed_x_etna_1/flow_searaft";
	//char const* const flow_2_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/zed_x_etna_1/flow_2_searaft";
	//char const* const disp_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/zed_x_etna_1/disp_searaft";
	//char const* const poses_path = "C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/zed_x_etna_1/pose";
	//char const* pattern = "left%06d.flo";


// hl2_to_opencv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)

//std::unique_ptr<Eigen::Matrix<float, 4, 4>[]> gt_poses = std::make_unique<Eigen::Matrix<float, 4, 4>[]>(N + 1);
//std::unique_ptr<Eigen::Matrix<float, 4, 4>[]> gt_relposes = std::make_unique<Eigen::Matrix<float, 4, 4>[]>(N);
//for (int i = 0; i < N; ++i)
		//{
		//	sprintf(path, "%s/%06d.flo", flow_path, fid + i);
		//	load_flow(path, flows_pt.get(), i, nullptr, nullptr, false, w, h);
		//}
		//for (int i = 0; i < (N - 1); ++i)
		//{
		//	sprintf(path, "%s/%06d.flo", flow_2_path, fid + i);
		//	load_flow(path, flows_2_pt.get(), i, nullptr, nullptr, false, w, h);
		//}
		//sprintf(path, "%s/%06d.flo", disp_path, fid);
		//load_flow(path, disparity_pt.get(), 0, nullptr, nullptr, false, w, h);
		//for (int i = 0; i < (N + 1); ++i)
		//{
		//	sprintf(path, "%s/%06d.flo", disp_path, fid + i);
		//	load_flow(path, disparities_pt.get(), i, nullptr, nullptr, false, w, h);
		//}
		//for (int i = 0; i < (N + 1); ++i)
		//{
		//	sprintf(path, "%s/%06d.bin", poses_path, fid + i);
		//	load_pose_hl2ss(path, gt_poses[i].data());
		//}
		//for (int i = 0; i < N; ++i)
		//{
		//	gt_relposes[i] = gt_poses[i + 1].inverse() * gt_poses[i];
		//}
//gt_poses[i] = hl2_to_opencv * load_pose(path).transpose() * hl2_to_opencv;

//sprintf(path, "%s/left%06d.flo", flow_path, fid + i);
			//sprintf(path, "%s/left%06d.flo", flow_2_path, fid + i);
		//sprintf(path, "%s/left%06d.flo", disp_path, fid);
			//sprintf(path, "%s/left%06d.flo", disp_path, fid + i);

//load_file(path, disparities_pt.get() + i * (w * h * 2), 12, -1);
			//load_file(path, disparity_pt.get(), 12, -1);
		//load_file(path, flows_pt.get() + i * (w * h * 2), 12, -1);
		//load_file(path, flows_2_pt.get() + i * (w * h * 2), 12, -1);

/*
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
*/