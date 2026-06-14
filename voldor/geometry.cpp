
#include <chrono>

#include "geometry.h"
#include "batch_solvers.h"
#include "helpers_opencv.h"
#include "helpers_camera.h"

#include "../gpu-kernels/gpu_kernels.h"

// OK
static void collect_point_correspondences(std::vector<cv::Mat> const& flows_1, std::vector<cv::Mat> const& flows_2, std::vector<cv::Mat> const& disparities, std::vector<cv::Mat> const& rigidnesses, cv::Mat const& depth, std::vector<Camera> const& cameras, int batch_flows, int active_index, bool update_batch_instance, bool update_loop_instance, Config const& options, std::vector<cv::Point3f>& p3d_1, std::vector<cv::Point2f>& p2k_2, std::vector<cv::Point3f>& p2z_1, std::vector<cv::Point3f>& p2z_2, std::vector<cv::Point3f>& p2z_3, std::vector<float>& tf_squared_error)
{
	bool tf_enable = options.multiview_mode == 3;

	int w = flows_1[0].cols;
	int h = flows_1[0].rows;
	
	int pixels = w * h;
	
	std::unique_ptr<float const* []> h_flows_1;
	std::unique_ptr<float const* []> h_flows_2;
	std::unique_ptr<float const* []> h_disparities;

	if (flows_1.size() > 0)
	{		
		h_flows_1 = std::make_unique<float const* []>(flows_1.size());
		for (int i = 0; i < flows_1.size(); ++i) { h_flows_1[i] = reinterpret_cast<float*>(flows_1[i].data); }
	}

	if (flows_2.size() > 0)
	{		
		h_flows_2 = std::make_unique<float const* []>(flows_2.size());
		for (int i = 0; i < flows_2.size(); ++i) { h_flows_2[i] = reinterpret_cast<float*>(flows_2[i].data); }
	}

	if (disparities.size() > 0)
	{		
		h_disparities = std::make_unique<float const* []>(disparities.size());
		for (int i = 0; i < disparities.size(); ++i) { h_disparities[i] = reinterpret_cast<float*>(disparities[i].data); }
	}

	std::unique_ptr<float const* []> h_rigidnesses = std::make_unique<float const* []>(batch_flows);
	std::unique_ptr<float const* []> h_Rs = std::make_unique<float const* []>(batch_flows);
	std::unique_ptr<float const* []> h_ts = std::make_unique<float const* []>(batch_flows);

	for (int i = 0; i < batch_flows; ++i)
	{ 
		h_rigidnesses[i] = reinterpret_cast<float*>(rigidnesses[i].data);
		h_Rs[i] = reinterpret_cast<float*>(cameras[i].R.data);
		h_ts[i] = reinterpret_cast<float*>(cameras[i].t.data);
	}

	float const* h_depth = reinterpret_cast<float*>(depth.data);
	float const* h_K = reinterpret_cast<float*>(cameras[active_index].K.data);

	p3d_1.resize(pixels);
	p2k_2.resize(pixels);
	p2z_1.resize(pixels);
	p2z_2.resize(pixels);
	p2z_3.resize(pixels);

	tf_squared_error.resize(pixels);

	float* h_p3d_1 = reinterpret_cast<float*>(p3d_1.data());
	float* h_p2k_2 = reinterpret_cast<float*>(p2k_2.data());
	float* h_p2z_1 = reinterpret_cast<float*>(p2z_1.data());
	float* h_p2z_2 = reinterpret_cast<float*>(p2z_2.data());
	float* h_p2z_3 = reinterpret_cast<float*>(p2z_3.data());

	float* h_tf_squared_error = tf_squared_error.data();

	if      (update_batch_instance) { collect_p3p_instances(h_flows_1.get(), h_flows_2.get(), h_disparities.get(), h_rigidnesses.get(), h_depth, h_K, h_Rs.get(), h_ts.get(), batch_flows, w, h, active_index, options.rigidness_threshold, options.rigidness_sum_threshold, options.pose_sample_min_depth, options.pose_sample_max_depth, options.max_trace_on_flow, options.disparities_enable, options.disparities_use_0, tf_enable, options.tf_enable_flow_2, options.tf_use_flow_2, options.tf_squared_error_threshold, h_p3d_1, h_p2k_2, h_p2z_1, h_p2z_2, h_p2z_3, h_tf_squared_error); }
	else if (update_loop_instance)  { collect_p3p_instances(nullptr,         nullptr,         nullptr,             h_rigidnesses.get(), h_depth, h_K, h_Rs.get(), h_ts.get(), batch_flows, w, h, active_index, options.rigidness_threshold, options.rigidness_sum_threshold, options.pose_sample_min_depth, options.pose_sample_max_depth, options.max_trace_on_flow, options.disparities_enable, options.disparities_use_0, tf_enable, options.tf_enable_flow_2, options.tf_use_flow_2, options.tf_squared_error_threshold, h_p3d_1, h_p2k_2, h_p2z_1, h_p2z_2, h_p2z_3, h_tf_squared_error); }
	else                            { collect_p3p_instances(nullptr,         nullptr,         nullptr,             nullptr,             nullptr, h_K, h_Rs.get(), h_ts.get(), batch_flows, w, h, active_index, options.rigidness_threshold, options.rigidness_sum_threshold, options.pose_sample_min_depth, options.pose_sample_max_depth, options.max_trace_on_flow, options.disparities_enable, options.disparities_use_0, tf_enable, options.tf_enable_flow_2, options.tf_use_flow_2, options.tf_squared_error_threshold, h_p3d_1, h_p2k_2, h_p2z_1, h_p2z_2, h_p2z_3, h_tf_squared_error); }

	int bf_count = 0;
	int tf_count = 0;

	for (int i = 0; i < pixels; ++i)
	{
		if (is_valid_point(p3d_1[i]) && is_valid_point(p2k_2[i]))
		{
			p3d_1[bf_count] = p3d_1[i];
			p2k_2[bf_count] = p2k_2[i];

			bf_count++;
		}

		if (is_valid_point(p2z_1[i]) && is_valid_point(p2z_2[i]) && (!tf_enable || is_valid_point(p2z_3[i])))
		{
			p2z_1[tf_count] = p2z_1[i];
			p2z_2[tf_count] = p2z_2[i];
			p2z_3[tf_count] = p2z_3[i];

			tf_squared_error[tf_count] = tf_squared_error[i];

			tf_count++;
		}
	}

	p3d_1.resize(bf_count);
	p2k_2.resize(bf_count);
	
	p2z_1.resize(tf_count);
	p2z_2.resize(tf_count);
	p2z_3.resize(tf_count);

	tf_squared_error.resize(tf_count);
}

// OK
static int solve_pose_pool(cv::Mat const& K, std::vector<cv::Point3f> const& p3d_1, std::vector<cv::Point2f> const& p2k_2, std::vector<cv::Point3f> const& p2z_1, std::vector<cv::Point3f> const& p2z_2, std::vector<cv::Point3f> const& p2z_3, Config const& options, cv::Mat& poses_pool, cv::Mat& velocities_pool, cv::Mat& focals_pool, cv::Mat& next_pool, int& next_pool_used)
{
	cv::Point3f const* p3d_1_data = p3d_1.data();
	cv::Point2f const* p2k_2_data = p2k_2.data();

	cv::Point3f const* p2z_1_data = p2z_1.data();
	cv::Point3f const* p2z_2_data = p2z_2.data();
	cv::Point3f const* p2z_3_data = p2z_3.data();

	int bf_count = static_cast<int>(p3d_1.size());
	int tf_count = static_cast<int>(p2z_1.size());

	bool set_velocities_pool = false;
	bool set_next_pool = false;
	bool set_focals_pool = false;

	switch (options.solver_select)
	{
	case 16: set_velocities_pool = true; break;
	case 17: set_velocities_pool = true; break;
	case 18: set_velocities_pool = true; break;
	case 24: set_next_pool = options.tf_enable_next_pool; break;
	case 25: set_next_pool = options.tf_enable_next_pool; break;
	case 32: set_focals_pool = true; break;
	case 33: set_velocities_pool = set_focals_pool = true; break;
	}

	float* poses_pool_data = nullptr;
	float* velocities_pool_data = nullptr;
	float* focals_pool_data = nullptr;
	float* next_pool_data = nullptr;
	int next_pool_pushed = 0;

	if (set_velocities_pool)
	{
		velocities_pool = cv::Mat(options.n_poses_to_sample, 6, CV_32F);
		velocities_pool_data = reinterpret_cast<float*>(velocities_pool.data);
	}
	else
	{
		velocities_pool_data = nullptr;
	}

	if (set_focals_pool) 
	{
		focals_pool = cv::Mat(options.n_poses_to_sample, 2, CV_32F);
		focals_pool_data = reinterpret_cast<float*>(focals_pool.data);
	}
	else
	{
		focals_pool_data = nullptr;
	}

	if (set_next_pool && (next_pool.total() <= 0))
	{
		next_pool = cv::Mat(1 * options.n_poses_to_sample, 6, CV_32F);
		next_pool_used = 0;
	}

	if (set_next_pool)
	{
		poses_pool = cv::Mat(2 * options.n_poses_to_sample, 6, CV_32F);
		memcpy(poses_pool.data, next_pool.data, sizeof(cv::Vec6f) * next_pool_used);
		poses_pool_data = reinterpret_cast<float*>(poses_pool.data) + (6 * next_pool_used);
		next_pool_data = reinterpret_cast<float*>(next_pool.data);
		next_pool_pushed = next_pool_used;
	}
	else
	{
		poses_pool = cv::Mat(1 * options.n_poses_to_sample, 6, CV_32F);
		poses_pool_data = reinterpret_cast<float*>(poses_pool.data);
		next_pool_data = nullptr;
		next_pool_pushed = 0;
	}

	int poses_pool_used = 0;

	switch (options.solver_select)
	{
	// p4p
	case  0: poses_pool_used = batch_cpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case  1: poses_pool_used = batch_cpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case  2: poses_pool_used = batch_gpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data); break;
	case  3: poses_pool_used = batch_gpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data); break;

	// gpm
	case  8: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 0, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case  9: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 1, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 10: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 2, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 11: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 3, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 12: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 4, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 13: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 5, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 14: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 6, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;
	case 15: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 7, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.batch_unique); break;

	// rnp
	case 16: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.batch_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;
	case 17: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.batch_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;
	case 18: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 2, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.batch_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;

	// tft
	case 24: poses_pool_used = batch_cpu_solver_tft(p2z_1_data, p2z_2_data, p2z_3_data, tf_count, K, 0, options.n_poses_to_sample, poses_pool_data, next_pool_data, options.batch_workers, options.batch_unique, options.tf_threshold, options.tf_sample_size); break;
	case 25: poses_pool_used = batch_cpu_solver_tft(p2z_1_data, p2z_2_data, p2z_3_data, tf_count, K, 1, options.n_poses_to_sample, poses_pool_data, next_pool_data, options.batch_workers, options.batch_unique, options.tf_threshold, options.tf_sample_size); break;
	case 26: poses_pool_used = batch_cpu_solver_tft(p2z_1_data, p2z_2_data, p2z_3_data, tf_count, K, 2, options.n_poses_to_sample, poses_pool_data, next_pool_data, options.batch_workers, options.batch_unique, options.tf_threshold, options.tf_sample_size); break;

	// p4pf
	case 32: poses_pool_used = batch_cpu_solver_ppf(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, focals_pool_data, options.batch_workers, options.batch_unique, options.square_pixels); break;
	case 33: poses_pool_used = batch_cpu_solver_rpf(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, focals_pool_data, options.batch_workers, options.batch_unique, options.square_pixels, options.rs_direction, options.rs_r0, options.rs_iterations); break;

	// default: gpu p4p lambdatwist
	default: poses_pool_used = batch_gpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data); break;
	}

	if (set_next_pool)
	{ 
		next_pool_used = poses_pool_used;
	}
	else
	{
		next_pool_used = 0;
	}

	return next_pool_pushed + poses_pool_used;
}

int optimize_camera_pose(std::vector<cv::Mat> const& flows_1, std::vector<cv::Mat> const& flows_2, std::vector<cv::Mat> const& disparities, std::vector<cv::Mat> const& rigidnesses, cv::Mat const& depth, std::vector<Camera>& cameras, int batch_flows, int active_index, bool successive_pose, bool rg_refine, bool update_batch_instance, bool update_loop_instance, Config const& options, cv::Mat& next_pool, int& next_pool_used)
{
	int const w = flows_1[0].cols;
	int const h = flows_1[0].rows;

	//----------------------------------------------------------------------------

	auto time_stamp = std::chrono::high_resolution_clock::now();

	std::vector<cv::Point2f> p2d_2;
	std::vector<cv::Point3f> p3d_1;
	std::vector<cv::Point3f> p2z_1;
	std::vector<cv::Point3f> p2z_2;
	std::vector<cv::Point3f> p2z_3;
	std::vector<float> trifocal_squared_error;

	collect_point_correspondences(flows_1, flows_2, disparities, rigidnesses, depth, cameras, batch_flows, active_index, update_batch_instance, update_loop_instance, options, p3d_1, p2d_2, p2z_1, p2z_2, p2z_3, trifocal_squared_error);

	if (!options.silent) { std::cout << "sampling collection time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;	}

	//----------------------------------------------------------------------------

	time_stamp = std::chrono::high_resolution_clock::now();

	cv::Mat poses_pool;
	cv::Mat velocities_pool;
	cv::Mat focals_pool;

	int poses_pool_used = solve_pose_pool(cameras[active_index].K, p3d_1, p2d_2, p2z_1, p2z_2, p2z_3, options, poses_pool, velocities_pool, focals_pool, next_pool, next_pool_used);
	if (poses_pool_used <= 0) { return 0; }

	poses_pool = poses_pool.rowRange(0, poses_pool_used);
	if (velocities_pool.total() > 0) { velocities_pool = velocities_pool.rowRange(0, poses_pool_used); }
	if (focals_pool.total() > 0) { focals_pool = focals_pool.rowRange(0, poses_pool_used); }

	if (!options.silent) { std::cout << "solver computing time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl; }

	//----------------------------------------------------------------------------

	cameras[active_index].pose_sample_count = poses_pool_used;

	cv::Mat pose_opm(1, 6, CV_32F);
	cv::Mat velocity_opm(1, 6, CV_32F);
	cv::Mat focal_opm(1, 2, CV_32F);

	Rodrigues(cameras[active_index].R, pose_opm.at<cv::Vec3f>(0));
	pose_opm.at<cv::Vec3f>(1) = cameras[active_index].t.at<cv::Vec3f>(0);

	if (velocities_pool.total() > 0)
	{
		velocity_opm.at<cv::Vec3f>(0) = cameras[active_index].dr.at<cv::Vec3f>(0);
		velocity_opm.at<cv::Vec3f>(1) = cameras[active_index].dt.at<cv::Vec3f>(1);
	}

	if (focals_pool.total() > 0)
	{
		focal_opm.at<float>(0) = cameras[active_index].K.at<float>(0, 0);
		focal_opm.at<float>(1) = cameras[active_index].K.at<float>(1, 1);
	}

	//----------------------------------------------------------------------------
	// scale and do meanshift

	// pose: 6 | 3+3
	// pose&velocity: 12 | 6+6 | 3+3+3+3
	// pose&focal: 7 | 6+1 | 3+3+1
	// pose&velocity&focal | 13 | 6+6+1 | 3+3+3+3+1
		
	time_stamp = std::chrono::high_resolution_clock::now();

	poses_pool.colRange(0, 3) *= options.meanshift_rvec_scale; // depends on translation scale?
	pose_opm.colRange(0, 3) *= options.meanshift_rvec_scale;
	meanshift_gpu
	(
		(float*)poses_pool.data,
		options.meanshift_kernel_var,
		(float*)pose_opm.data,
		&cameras[active_index].pose_density,
		&cameras[active_index].last_used_ms_iters,
		successive_pose,
		poses_pool_used,
		6,
		options.meanshift_epsilon,
		options.meanshift_max_iters,
		options.meanshift_max_init_trials,
		options.meanshift_good_init_confidence
	);

	if (focals_pool.total() > 0)
	{
		float focals_scale = (25.0f / static_cast<float>(std::sqrt(w * w + h * h)));
		focals_pool *= focals_scale;
		focal_opm *= focals_scale;
		meanshift_gpu
		(
			(float*)focals_pool.data,
			options.meanshift_kernel_var,
			(float*)focal_opm.data,
			nullptr,
			nullptr,
			successive_pose,
			poses_pool_used,
			2,
			options.meanshift_epsilon,
			options.meanshift_max_iters,
			options.meanshift_max_init_trials,
			options.meanshift_good_init_confidence
		);
		focal_opm /= focals_scale;
	}

	if (!options.silent) { std::cout << "meanshift time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl; }

	//-------------------------------------------------------------------------

	if (rg_refine) {
		time_stamp = std::chrono::high_resolution_clock::now();

		cameras[active_index].pose_covar = 0;
		for (int d = 0; d < 6; d++)
			cameras[active_index].pose_covar.at<float>(d, d) = options.meanshift_kernel_var;

		cameras[active_index].pose_covar *= (options.rg_pose_scaling* options.rg_pose_scaling);
		pose_opm *= options.rg_pose_scaling;
		poses_pool *= options.rg_pose_scaling;

		int fit_rg_ret = fit_robust_gaussian((float*)poses_pool.data, (float*)pose_opm.data, (float*)cameras[active_index].pose_covar.data,
			options.rg_trunc_sigma, options.rg_covar_reg_lambda, &cameras[active_index].pose_density, &cameras[active_index].last_used_gu_iters, poses_pool_used, 6, options.rg_epsilon, options.rg_max_iters);

		if (fit_rg_ret == 0) { // cudaSuccess==0 
			cameras[active_index].pose_covar /= (options.rg_pose_scaling*options.rg_pose_scaling);
			for (int i1 = 0; i1 < 6; i1++) {
				for (int i2 = 0; i2 < 6; i2++) {
					if (i1 < 3 || i2 < 3)
						cameras[active_index].pose_covar.at<float>(i1, i2) /= options.meanshift_rvec_scale;
					if (i1 < 3 && i2 < 3)
						cameras[active_index].pose_covar.at<float>(i1, i2) /= options.meanshift_rvec_scale;
				}
			}

		}
		else {
			cameras[active_index].pose_covar = 0;
		}

		pose_opm /= options.rg_pose_scaling;
		//poses_pool /= cfg.rg_pose_scaling; // this is not used later, no need scale back

		if (!options.silent) { std::cout << "gu fit time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl; }
	}

	pose_opm.colRange(0, 3) /= options.meanshift_rvec_scale;
	//poses_pool.colRange(0, 3) /= cfg.meanshift_rvec_scale; // this is not used later, no need scale back

	time_stamp = std::chrono::high_resolution_clock::now();

	if (cv::checkRange(pose_opm)) { //check if gpu gives nan...
		// copy back pose
		Rodrigues(pose_opm.at<cv::Vec3f>(0), cameras[active_index].R);
		cameras[active_index].t.at<cv::Vec3f>(0) = pose_opm.at<cv::Vec3f>(1);

		if (focals_pool.total() > 0)
		{
			if (!cv::checkRange(focal_opm)) { return 0; }

			for (int index = options.shared_focals ? 0 : active_index; index <= active_index; ++index)
			{
				cameras[index].K.at<float>(0, 0) = focal_opm.at<float>(0);
				cameras[index].K.at<float>(1, 1) = focal_opm.at<float>(1);
				cameras[index].K_inv = cameras[index].K.inv();
			}
			
		}

		return 1;
	}
	else
		return 0;

}
