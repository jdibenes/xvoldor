
#include "geometry.h"
#include "../gpu-kernels/gpu_kernels.h"
#include "solvers.h"
#include "batch_solver.h"
#include "helpers_opencv.h"







static
void
collect_p3p_correspondences
(
	std::vector<cv::Mat> const& flows_1,
	std::vector<cv::Mat> const& flows_2,
	std::vector<cv::Mat> const& disparities,
	std::vector<cv::Mat> const& rigidnesses,
	cv::Mat const& depth,
	std::vector<Camera> const& cams,
	int n_flows,
	int active_idx,
	bool update_batch_instance,
	bool update_iter_instance,
	Config const& cfg,
	std::vector<cv::Point2f>& p2k_2,
	std::vector<cv::Point3f>& p3d_1,
	std::vector<cv::Point3f>& p2z_1,
	std::vector<cv::Point3f>& p2z_2,
	std::vector<cv::Point3f>& p2z_3,
	std::vector<float>& tf_squared_error
)
{
	int const w = flows_1[0].cols;
	int const h = flows_1[0].rows;

	float const** h_flows_1 = NULL;
	float const** h_flows_2 = NULL;
	float const** h_disparities = NULL;
	float const** h_rigidnesses = NULL;
	float const** h_Rs = NULL;
	float const** h_ts = NULL;
	
	if (flows_1.size() > 0)
	{
		h_flows_1 = new float const* [flows_1.size()];
		for (int i = 0; i < flows_1.size(); ++i) { h_flows_1[i] = (float*)flows_1[i].data; }
	}

	if (flows_2.size() > 0)
	{
		h_flows_2 = new float const* [flows_2.size()];
		for (int i = 0; i < flows_2.size(); ++i) { h_flows_2[i] = (float*)flows_2[i].data; }
	}

	if (disparities.size() > 0)
	{
		h_disparities = new float const* [disparities.size()];
		for (int i = 0; i < disparities.size(); ++i) { h_disparities[i] = (float*)disparities[i].data; }
	}

	h_rigidnesses = new float const* [n_flows];
	h_Rs = new float const* [n_flows];
	h_ts = new float const* [n_flows];

	for (int i = 0; i < n_flows; ++i)
	{ 
		h_rigidnesses[i] = (float*)rigidnesses[i].data;
		h_Rs[i] = (float*)cams[i].R.data;
		h_ts[i] = (float*)cams[i].t.data;
	}

	p2k_2.resize(w * h);
	p3d_1.resize(w * h);

	p2z_1.resize(w * h);
	p2z_2.resize(w * h);
	p2z_3.resize(w * h);

	tf_squared_error.resize(w * h);

	if (update_batch_instance) {
		collect_p3p_instances
		(
			h_flows_1,
			h_rigidnesses,
			(float*)depth.data,
			(float*)cams[0].K.data,
			h_Rs,
			h_ts,
			(float*)p2k_2.data(),
			(float*)p3d_1.data(),
			n_flows,
			w,
			h,
			active_idx,
			cfg.rigidness_threshold,
			cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth,
			cfg.pose_sample_max_depth,
			cfg.max_trace_on_flow,
			h_flows_2,
			(float*)p2z_1.data(),
			(float*)p2z_2.data(),
			(float*)p2z_3.data(),
			h_disparities,
			tf_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.tf_index_0,
			cfg.tf_index_1,
			cfg.tf_index_2,
			cfg.tf_enable_flow_2,
			cfg.tf_squared_error_min_thresh,
			cfg.tf_squared_error_max_thresh
		);
	}
	else if (update_iter_instance) {
		collect_p3p_instances
		(
			NULL,
			h_rigidnesses,
			(float*)depth.data,
			NULL,
			h_Rs,
			h_ts,
			(float*)p2k_2.data(),
			(float*)p3d_1.data(),
			n_flows,
			w,
			h,
			active_idx,
			cfg.rigidness_threshold,
			cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth,
			cfg.pose_sample_max_depth,
			cfg.max_trace_on_flow,
			NULL,
			(float*)p2z_1.data(),
			(float*)p2z_2.data(),
			(float*)p2z_3.data(),
			NULL,
			tf_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.tf_index_0,
			cfg.tf_index_1,
			cfg.tf_index_2,
			cfg.tf_enable_flow_2,
			cfg.tf_squared_error_min_thresh,
			cfg.tf_squared_error_max_thresh
		);
	}
	else {
		collect_p3p_instances
		(
			NULL,
			NULL,
			NULL,
			NULL,
			h_Rs,
			h_ts,
			(float*)p2k_2.data(),
			(float*)p3d_1.data(),
			n_flows,
			w,
			h,
			active_idx,
			cfg.rigidness_threshold,
			cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth,
			cfg.pose_sample_max_depth,
			cfg.max_trace_on_flow,
			NULL,
			(float*)p2z_1.data(),
			(float*)p2z_2.data(),
			(float*)p2z_3.data(),
			NULL,
			tf_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.tf_index_0,
			cfg.tf_index_1,
			cfg.tf_index_2,
			cfg.tf_enable_flow_2,
			cfg.tf_squared_error_min_thresh,
			cfg.tf_squared_error_max_thresh
		);
	}

	if (h_flows_1) { delete[] h_flows_1; }
	if (h_flows_2) { delete[] h_flows_2; }
	if (h_disparities) { delete[] h_disparities; }
	if (h_rigidnesses) { delete[] h_rigidnesses; }
	if (h_Rs) { delete[] h_Rs; }
	if (h_ts) { delete[] h_ts; }
	

	int bf_count = 0;
	int tf_count = 0;

	for (int i = 0; i < (w * h); ++i)
	{
		if (is_valid_point(p3d_1[i]) && is_valid_point(p2k_2[i]))
		{
			p3d_1[bf_count] = p3d_1[i];
			p2k_2[bf_count] = p2k_2[i];

			bf_count++;
		}

		if (is_valid_point(p2z_1[i]) && is_valid_point(p2z_2[i]) && is_valid_point(p2z_3[i]))
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
		focals_pool = cv::Mat(options.n_poses_to_sample, 1, CV_32F);
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
	case  0: poses_pool_used = batch_cpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case  1: poses_pool_used = batch_cpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case  2: poses_pool_used = batch_gpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data); break;
	case  3: poses_pool_used = batch_gpu_solver_p4p(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data); break;

	// gpm
	case  8: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 0, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case  9: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 1, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 10: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 2, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 11: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 3, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 12: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 4, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 13: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 5, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 14: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 6, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;
	case 15: poses_pool_used = batch_cpu_solver_gpm(p2z_1_data, p2z_2_data, tf_count, K, 7, options.n_poses_to_sample, poses_pool_data, options.batch_workers, options.sample_unique); break;

	// rnp
	case 16: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 0, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.sample_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;
	case 17: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 1, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.sample_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;
	case 18: poses_pool_used = batch_cpu_solver_rnp(p3d_1_data, p2k_2_data, bf_count, K, 2, options.n_poses_to_sample, poses_pool_data, velocities_pool_data, options.batch_workers, options.sample_unique, options.rs_direction, options.rs_r0, options.rs_iterations); break;

	// tft
	case 24: poses_pool_used = batch_cpu_solver_tft(p2z_1_data, p2z_2_data, p2z_3_data, tf_count, K, 0, options.n_poses_to_sample, poses_pool_data, next_pool_data, options.batch_workers, options.sample_unique, options.tf_threshold); break;

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

































int 
optimize_camera_pose
(
	std::vector<cv::Mat> const& flows,
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
	std::vector<cv::Mat> const& flows_2,
	std::vector<cv::Mat> const& disparities,
	cv::Mat& next_pool,
	int& next_pool_used
) 
{

	int const w = flows[0].cols;
	int const h = flows[0].rows;

	//----------------------------------------------------------------------------

	auto time_stamp = std::chrono::high_resolution_clock::now();

	std::vector<cv::Point2f> p2d_2;
	std::vector<cv::Point3f> p3d_1;
	std::vector<cv::Point3f> p2z_1;
	std::vector<cv::Point3f> p2z_2;
	std::vector<cv::Point3f> p2z_3;
	std::vector<float> trifocal_squared_error;

	collect_p3p_correspondences(flows, flows_2, disparities, rigidnesses, depth, cams, n_flows, active_idx, update_batch_instance, update_iter_instance, cfg, p2d_2, p3d_1, p2z_1, p2z_2, p2z_3, trifocal_squared_error);

	if (!cfg.silent) { std::cout << "sampling collection time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;	}

	//----------------------------------------------------------------------------

	time_stamp = std::chrono::high_resolution_clock::now();

	cv::Mat poses_pool;
	cv::Mat velocities_pool;
	cv::Mat focals_pool;

	int poses_pool_used = solve_pose_pool(cams[active_idx].K, p3d_1, p2d_2, p2z_1, p2z_2, p2z_3, cfg, poses_pool, velocities_pool, focals_pool, next_pool, next_pool_used);
	if (poses_pool_used <= 0) { return 0; }

	if (!cfg.silent) { std::cout << "solver computing time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl; }

	//----------------------------------------------------------------------------





	cams[active_idx].pose_sample_count = poses_pool_used;

	poses_pool = poses_pool.rowRange(0, poses_pool_used);
	
	cv::Mat pose_opm(1, 6, CV_32F);
	Rodrigues(cams[active_idx].R, pose_opm.at<cv::Vec3f>(0));
	pose_opm.at<cv::Vec3f>(1) = cams[active_idx].t.at<cv::Vec3f>(0);

	if (velocities_pool.total() > 0)
	{
		velocities_pool = velocities_pool.rowRange(0, poses_pool_used);

		cv::Mat velocity_opm(1, 6, CV_32F);
		velocity_opm.at<cv::Vec3f>(0) = cams[active_idx].dr.at<cv::Vec3f>(0);
		velocity_opm.at<cv::Vec3f>(1) = cams[active_idx].dt.at<cv::Vec3f>(1);
	}

	if (focals_pool.total() > 0)
	{
		focals_pool = focals_pool.rowRange(0, poses_pool_used);

		float focal_opm = cams[active_idx].focal;
	}

	//----------------------------------------------------------------------------
	// scale and do meanshift

	// pose: 6 | 3+3
	// pose&velocity: 12 | 6+6 | 3+3+3+3
	// pose&focal: 7 | 6+1 | 3+3+1








	
	time_stamp = std::chrono::high_resolution_clock::now();

	///*
	poses_pool.colRange(0, 3) *= cfg.meanshift_rvec_scale; // depends on translation scale?
	pose_opm.colRange(0, 3) *= cfg.meanshift_rvec_scale;
	meanshift_gpu
	(
		(float*)poses_pool.data,
		cfg.meanshift_kernel_var,
		(float*)pose_opm.data,
		&cams[active_idx].pose_density,
		&cams[active_idx].last_used_ms_iters,
		successive_pose,
		poses_pool_used,
		6,
		cfg.meanshift_epsilon,
		cfg.meanshift_max_iters,
		cfg.meanshift_max_init_trials,
		cfg.meanshift_good_init_confidence
	);
	//*/



	if (!cfg.silent)
	{
		std::cout << "meanshift time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
	}


	//-------------------------------------------------------------------------

	if (rg_refine) {
		time_stamp = std::chrono::high_resolution_clock::now();

		cams[active_idx].pose_covar = 0;
		for (int d = 0; d < 6; d++)
			cams[active_idx].pose_covar.at<float>(d, d) = cfg.meanshift_kernel_var;

		cams[active_idx].pose_covar *= (cfg.rg_pose_scaling* cfg.rg_pose_scaling);
		pose_opm *= cfg.rg_pose_scaling;
		poses_pool *= cfg.rg_pose_scaling;

		/*
		int rvec_zero_count = 0;
		for (int i = 0; i < poses_pool_used; i++)
			if (poses_pool.at<float>(i, 0) == 0)
				rvec_zero_count++;
		cout << "rvec_zero_count = " << rvec_zero_count << endl;
		*/


		int fit_rg_ret = fit_robust_gaussian((float*)poses_pool.data, (float*)pose_opm.data, (float*)cams[active_idx].pose_covar.data,
			cfg.rg_trunc_sigma, cfg.rg_covar_reg_lambda, &cams[active_idx].pose_density, &cams[active_idx].last_used_gu_iters, poses_pool_used, 6, cfg.rg_epsilon, cfg.rg_max_iters);

		if (fit_rg_ret == 0) { // cudaSuccess==0 
			cams[active_idx].pose_covar /= (cfg.rg_pose_scaling*cfg.rg_pose_scaling);
			for (int i1 = 0; i1 < 6; i1++) {
				for (int i2 = 0; i2 < 6; i2++) {
					if (i1 < 3 || i2 < 3)
						cams[active_idx].pose_covar.at<float>(i1, i2) /= cfg.meanshift_rvec_scale;
					if (i1 < 3 && i2 < 3)
						cams[active_idx].pose_covar.at<float>(i1, i2) /= cfg.meanshift_rvec_scale;
				}
			}

		}
		else {
			cams[active_idx].pose_covar = 0;
		}

		pose_opm /= cfg.rg_pose_scaling;
		//poses_pool /= cfg.rg_pose_scaling; // this is not used later, no need scale back


		if (!cfg.silent)
			std::cout << "gu fit time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
	}


	pose_opm.colRange(0, 3) /= cfg.meanshift_rvec_scale;
	//poses_pool.colRange(0, 3) /= cfg.meanshift_rvec_scale; // this is not used later, no need scale back




	time_stamp = std::chrono::high_resolution_clock::now();

	if (cv::checkRange(pose_opm)) { //check if gpu gives nan...
		// copy back pose
		Rodrigues(pose_opm.at<cv::Vec3f>(0), cams[active_idx].R);
		cams[active_idx].t.at<cv::Vec3f>(0) = pose_opm.at<cv::Vec3f>(1);
		return 1;
	}
	else
		return 0;

}

void
estimate_depth_closed_form
(
	cv::Mat flow,
	cv::Mat& depth,
	Camera cam,
	float min_depth,
	float max_depth
)
{
	cv::Mat b = cam.K * cam.t;
	cv::Mat KRKinv = cam.K * cam.R * cam.K_inv;

	float b1 = b.at<float>(0), b2 = b.at<float>(1), b3 = b.at<float>(2);
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			cv::Point2f delta = flow.at<cv::Point2f>(y, x);
			cv::Mat P = (cv::Mat_<float>(3, 1) << x, y, 1);
			P = KRKinv * P;
			float w1 = P.at<float>(0), w2 = P.at<float>(1), w3 = P.at<float>(2);
			float a1 = x + delta.x, a2 = y + delta.y;
			float z_nume = (a1 * b3 - b1)*(w1 - a1 * w3) + (a2 * b3 - b2)*(w2 - a2 * w3);
			float z_deno = (w1 - a1 * w3)*(w1 - a1 * w3) + (w2 - a2 * w3)*(w2 - a2 * w3);
			depth.at<float>(y, x) = fminf(fmaxf(z_nume / z_deno, min_depth), max_depth);
		}
	}
}


void
estimate_camera_pose_epipolar
(
	cv::Mat flow,
	Camera& cam,
	cv::Mat mask,
	int sampling_2d_step
)
{
	int w = flow.cols, h = flow.rows;
	bool use_external_mask = !mask.empty();

	cv::Mat pts1(w*h, 2, CV_32F);
	cv::Mat pts2(w*h, 2, CV_32F);
	float* pts1_iter = (float*)pts1.data;
	float* pts2_iter = (float*)pts2.data;
	int N_used = 0;
	for (int y = 0; y < h; y += sampling_2d_step) {
		for (int x = 0; x < w; x += sampling_2d_step) {
			if (use_external_mask && mask.at<float>(y, x) < 0.5)
				continue;
			N_used++;
			*pts1_iter++ = x;
			*pts1_iter++ = y;
			*pts2_iter++ = x + flow.at<cv::Vec2f>(y, x)[0];
			*pts2_iter++ = y + flow.at<cv::Vec2f>(y, x)[1];
		}
	}

	pts1 = pts1.rowRange(0, N_used);
	pts2 = pts2.rowRange(0, N_used);
	//cam.F = findFundamentalMat(pts1, pts2, CV_RANSAC);

	if (use_external_mask) {
		cv::Mat pts_mask;
		cam.E = findEssentialMat(pts1, pts2, cam.K, cv::LMEDS, 0.999, 1.0, pts_mask);
		//cam.E = findHomography(pts1, pts2, LMEDS, 3.0, pts_mask);
	}
	else {
		cv::Mat pts_mask;
		cam.E = findEssentialMat(pts1, pts2, cam.K, cv::LMEDS, 0.999, 1.0, pts_mask);
		//cam.E = findHomography(pts1, pts2, LMEDS, 3.0, pts_mask);
	}


	recoverPose(cam.E, pts1, pts2, cam.K, cam.R, cam.t);
	cam.E.convertTo(cam.E, CV_32F);
	cam.R.convertTo(cam.R, CV_32F);
	cam.t.convertTo(cam.t, CV_32F);
	cam.t = cam.R*cam.t;

}




//void apply_meanshift(cv::Mat& poses_pool, cv::Mat& velocities_pool, cv::Mat& focals_pool, Config const& options, cv::Mat& pose_opm, cv::Mat& velocity_opm, float focal_opm)
//{
	// pose: 6 | 3+3
	// pose&velocity: 12 | 6+6 | 3+3+3+3
	// pose&focal: 7 | 6+1 | 3+3+1
	// median algorithm?

	/*
	cv::Mat poses_pool_r = poses_pool.colRange(0, 3);
	cv::Mat poses_pool_t = poses_pool.colRange(3, 6);

	meanshift_gpu
	(
		reinterpret_cast<float*>(poses_pool_r.data),
		options.meanshift_kernel_var,
		reinterpret_cast<float*>(pose_opm.data),
		&cams[active_idx].pose_density,
		&cams[active_idx].last_used_ms_iters,
		successive_pose,
		poses_pool_r.rows,
		3,
		options.meanshift_epsilon,
		options.meanshift_max_iters,
		options.meanshift_max_init_trials,
		options.meanshift_good_init_confidence
	);
	*/


	/*
	poses_pool.colRange(0, 3) *= cfg.meanshift_rvec_scale; // depends on translation scale?
	pose_opm.colRange(0, 3) *= cfg.meanshift_rvec_scale;
	meanshift_gpu
	(
		(float*)poses_pool.data,
		cfg.meanshift_kernel_var,
		(float*)pose_opm.data,
		&cams[active_idx].pose_density,
		&cams[active_idx].last_used_ms_iters,
		successive_pose,
		poses_pool_used,
		6,
		cfg.meanshift_epsilon,
		cfg.meanshift_max_iters,
		cfg.meanshift_max_init_trials,
		cfg.meanshift_good_init_confidence
	);
	*/

	//}

/*
	int n_p3p_points = 0;
	int n_p3v_points = 0;

	cv::Point2f const* pts2_src = p2k_2.data();
	cv::Point3f const* pts3_src = p3d_1.data();
	cv::Point3f const* p3v0_src = p2z_1.data();
	cv::Point3f const* p3v1_src = p2z_2.data();
	cv::Point3f const* p3v2_src = p2z_3.data();
	float const* tse2_src = trifocal_squared_error.data();

	cv::Point2f* pts2_dst = p2k_2.data();
	cv::Point3f* pts3_dst = p3d_1.data();
	cv::Point3f* p3v0_dst = p2z_1.data();
	cv::Point3f* p3v1_dst = p2z_2.data();
	cv::Point3f* p3v2_dst = p2z_3.data();
	float* tse2_dst = trifocal_squared_error.data();

	for (int i = 0; i < w * h; i++)
	{
		float pts2_sum = pts2_src->x + pts2_src->y;
		float pts3_sum = pts3_src->x + pts3_src->y + pts3_src->z;

		float ptsX_sum = pts2_sum + pts3_sum;

		float p3v0_sum = p3v0_src->x + p3v0_src->y + p3v0_src->z;
		float p3v1_sum = p3v1_src->x + p3v1_src->y + p3v1_src->z;
		float p3v2_sum = p3v2_src->x + p3v2_src->y + p3v2_src->z;

		float p3vX_sum = p3v0_sum + p3v1_sum + ((cfg.multiview_mode == 3) ? p3v2_sum : 0.0f);

		if (isfinite(ptsX_sum) && isfinite(p3vX_sum))
		{
			pts2_dst[n_p3p_points] = *pts2_src;
			pts3_dst[n_p3p_points] = *pts3_src;

			p3v0_dst[n_p3v_points] = *p3v0_src;
			p3v1_dst[n_p3v_points] = *p3v1_src;
			p3v2_dst[n_p3v_points] = *p3v2_src;
			tse2_dst[n_p3v_points] = *tse2_src;

			n_p3v_points++;

			n_p3p_points++;
		}

		pts2_src++;
		pts3_src++;


		p3v0_src++;
		p3v1_src++;
		p3v2_src++;
		tse2_src++;
	}

	p2k_2.resize(n_p3p_points);
	p3d_1.resize(n_p3p_points);

	p2z_1.resize(n_p3v_points);
	p2z_2.resize(n_p3v_points);
	p2z_3.resize(n_p3v_points);

	trifocal_squared_error.resize(n_p3v_points);
	*/

	// pts2 is related to frame(active_idx)
		// pts3 is related to frame(active_idx-1)
		// Thus, the relative pose describe frame(active_idx-1)--[R|Rt]-->frame(active_idx).