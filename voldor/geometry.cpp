
#include "geometry.h"
#include "../gpu-kernels/gpu_kernels.h"
#include "../lambdatwist/lambdatwist_p4p.h"
#include "trifocal.h"
#include "batch_solve.h"
//#include "trifocal_poselib.h"


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
	std::vector<cv::Point2f>& pts2_map,
	std::vector<cv::Point3f>& pts3_map,
	std::vector<cv::Point3f>& trifocal_0_map,
	std::vector<cv::Point3f>& trifocal_1_map,
	std::vector<cv::Point3f>& trifocal_2_map,
	std::vector<float>& trifocal_squared_error
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

	pts2_map.resize(w * h);
	pts3_map.resize(w * h);

	trifocal_0_map.resize(w * h);
	trifocal_1_map.resize(w * h);
	trifocal_2_map.resize(w * h);

	trifocal_squared_error.resize(w * h);

	if (update_batch_instance) {
		collect_p3p_instances
		(
			h_flows_1,
			h_rigidnesses,
			(float*)depth.data,
			(float*)cams[0].K.data,
			h_Rs,
			h_ts,
			(float*)pts2_map.data(),
			(float*)pts3_map.data(),
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
			(float*)trifocal_0_map.data(),
			(float*)trifocal_1_map.data(),
			(float*)trifocal_2_map.data(),
			h_disparities,
			trifocal_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.trifocal_index_0,
			cfg.trifocal_index_1,
			cfg.trifocal_index_2,
			cfg.trifocal_enable_flow_2,
			cfg.trifocal_squared_error_min_thresh,
			cfg.trifocal_squared_error_max_thresh
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
			(float*)pts2_map.data(),
			(float*)pts3_map.data(),
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
			(float*)trifocal_0_map.data(),
			(float*)trifocal_1_map.data(),
			(float*)trifocal_2_map.data(),
			NULL,
			trifocal_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.trifocal_index_0,
			cfg.trifocal_index_1,
			cfg.trifocal_index_2,
			cfg.trifocal_enable_flow_2,
			cfg.trifocal_squared_error_min_thresh,
			cfg.trifocal_squared_error_max_thresh
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
			(float*)pts2_map.data(),
			(float*)pts3_map.data(),
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
			(float*)trifocal_0_map.data(),
			(float*)trifocal_1_map.data(),
			(float*)trifocal_2_map.data(),
			NULL,
			trifocal_squared_error.data(),
			cfg.disparities_enable,
			cfg.multiview_mode == 3,
			cfg.trifocal_index_0,
			cfg.trifocal_index_1,
			cfg.trifocal_index_2,
			cfg.trifocal_enable_flow_2,
			cfg.trifocal_squared_error_min_thresh,
			cfg.trifocal_squared_error_max_thresh
		);
	}

	if (h_flows_1) { delete[] h_flows_1; }
	if (h_flows_2) { delete[] h_flows_2; }
	if (h_disparities) { delete[] h_disparities; }
	if (h_rigidnesses) { delete[] h_rigidnesses; }
	if (h_Rs) { delete[] h_Rs; }
	if (h_ts) { delete[] h_ts; }
	


	// pts2 is related to frame(active_idx)
	// pts3 is related to frame(active_idx-1)
	// Thus, the relative pose describe frame(active_idx-1)--[R|Rt]-->frame(active_idx).

	int n_p3p_points = 0;
	int n_p3v_points = 0;

	cv::Point2f const* pts2_src = pts2_map.data();
	cv::Point3f const* pts3_src = pts3_map.data();
	cv::Point3f const* p3v0_src = trifocal_0_map.data();
	cv::Point3f const* p3v1_src = trifocal_1_map.data();
	cv::Point3f const* p3v2_src = trifocal_2_map.data();
	float const* tse2_src = trifocal_squared_error.data();

	cv::Point2f* pts2_dst = pts2_map.data();
	cv::Point3f* pts3_dst = pts3_map.data();
	cv::Point3f* p3v0_dst = trifocal_0_map.data();
	cv::Point3f* p3v1_dst = trifocal_1_map.data();
	cv::Point3f* p3v2_dst = trifocal_2_map.data();
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

			//std::cout << "trifocal_error: " << *tse2_src << std::endl;

			n_p3v_points++;

			n_p3p_points++;
		}

		pts2_src++;
		pts3_src++;

		

		//if ()
		//{
			
		//}

		p3v0_src++;
		p3v1_src++;
		p3v2_src++;
		tse2_src++;
	}

	pts2_map.resize(n_p3p_points);
	pts3_map.resize(n_p3p_points);

	trifocal_0_map.resize(n_p3v_points);
	trifocal_1_map.resize(n_p3v_points);
	trifocal_2_map.resize(n_p3v_points);

	trifocal_squared_error.resize(n_p3v_points);
}



int
solve_p3p_pool
(
	std::vector<Camera>& cams,
	std::vector<cv::Point2f> const& pts2_map,
	std::vector<cv::Point3f> const& pts3_map,
	int active_idx,
	Config const& cfg,
	cv::Mat& poses_pool,
	std::vector<cv::Point3f> const& trifocal_0_map,
	std::vector<cv::Point3f> const& trifocal_1_map,
	std::vector<cv::Point3f> const& trifocal_2_map
)
{
	int poses_pool_used = 0;

	std::vector<cv::Vec6f>* next_pool = (active_idx < (cams.size() - 1)) ? &cams[active_idx + 1].trifocal_1_2_pool : NULL;

	//cams[active_idx + 1].trifocal_1_2_pool.clear(); // active_idx + 1 might not exist
	if (next_pool) { next_pool->clear(); }

	switch (cfg.solver_select)
	{
	case 0:  poses_pool_used = batch_solve_ap3p_cpu(pts2_map, pts3_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	case 1:  poses_pool_used = batch_solve_lambdatwist_cpu(pts2_map, pts3_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	case 2:  poses_pool_used = batch_solve_ap3p_gpu(pts2_map, pts3_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	case 3:  poses_pool_used = batch_solve_lambdatwist_gpu(pts2_map, pts3_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	
	case 5:  poses_pool_used = batch_solve_tft_linear_cpu(trifocal_0_map, trifocal_1_map, trifocal_2_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool, next_pool); break;

	case 4:  poses_pool_used = batch_solve_gpm_hpc0_cpu(trifocal_0_map, trifocal_1_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	case 6:  poses_pool_used = batch_solve_gpm_hpc1_cpu(trifocal_0_map, trifocal_1_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool, cfg.root_refine_interations); break;
	case 7:  poses_pool_used = batch_solve_gpm_hpc2_cpu(trifocal_0_map, trifocal_1_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool, cfg.root_refine_interations); break;


	default: poses_pool_used = batch_solve_lambdatwist_gpu(pts2_map, pts3_map, cams[active_idx].K, cfg.n_poses_to_sample, poses_pool); break;
	}

	for (cv::Vec6f const &v : cams[active_idx].trifocal_1_2_pool)
	{
		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = ((cv::Vec3f*)&v)[0];
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = ((cv::Vec3f*)&v)[1];

		poses_pool_used++;

		//std::cout << "active_idx: " << active_idx << " | " << v << std::endl;
	}

	return poses_pool_used;
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
	bool update_batch_instance,
	bool update_iter_instance,
	Config const& cfg,
	std::vector<cv::Mat> const& flows_2,
	std::vector<cv::Mat> const& disparities
) 
{

	int const w = flows[0].cols;
	int const h = flows[0].rows;

	//----------------------------------------------------------------------------

	auto time_stamp = std::chrono::high_resolution_clock::now();

	std::vector<cv::Point2f> pts2_map;
	std::vector<cv::Point3f> pts3_map;
	std::vector<cv::Point3f> trifocal_0_map;
	std::vector<cv::Point3f> trifocal_1_map;
	std::vector<cv::Point3f> trifocal_2_map;
	std::vector<float> trifocal_squared_error;

	collect_p3p_correspondences(flows, flows_2, disparities, rigidnesses, depth, cams, n_flows, active_idx, update_batch_instance, update_iter_instance, cfg, pts2_map, pts3_map, trifocal_0_map, trifocal_1_map, trifocal_2_map, trifocal_squared_error);

	if (!cfg.silent) { std::cout << "sampling collection time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;	}

	//----------------------------------------------------------------------------

	time_stamp = std::chrono::high_resolution_clock::now();

	cv::Mat poses_pool(2 * cfg.n_poses_to_sample, 6, CV_32F);
	int poses_pool_used = solve_p3p_pool(cams, pts2_map, pts3_map, active_idx, cfg, poses_pool, trifocal_0_map, trifocal_1_map, trifocal_2_map);

	if (poses_pool_used <= 0) { return 0; }

	if (!cfg.silent) { std::cout << "p3p computing time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl; }

	//----------------------------------------------------------------------------------

	time_stamp = std::chrono::high_resolution_clock::now(); // TODO: ???
	
	poses_pool = poses_pool.rowRange(0, poses_pool_used);
	cams[active_idx].pose_sample_count = poses_pool_used;

	cv::Mat pose_opm(1, 6, CV_32F);
	Rodrigues(cams[active_idx].R, pose_opm.at<cv::Vec3f>(0));
	pose_opm.at<cv::Vec3f>(1) = cams[active_idx].t.at<cv::Vec3f>(0);

	//----------------------------------------------------------------------------


	// scale and do meanshift
	time_stamp = std::chrono::high_resolution_clock::now();

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








//cv::Mat pts2_map(cv::Size(w, h), CV_32FC2);
//cv::Mat pts3_map(cv::Size(w, h), CV_32FC3);

//cv::Mat trifocal_map_0(cv::Size(w, h), CV_32FC3);
//cv::Mat trifocal_map_1(cv::Size(w, h), CV_32FC3);
//cv::Mat trifocal_map_2(cv::Size(w, h), CV_32FC3);


//cv::Mat trifocal_squared_error(cv::Size(w, h), CV_32F);

/*
std::cout << "P3P points: " << n_points << std::endl;
std::cout << "TRI (0)" << std::endl;
std::cout << tm0[0] << std::endl;
std::cout << "TRI (1)" << std::endl;
std::cout << tm1[0] << std::endl;
std::cout << "TRI (2)" << std::endl;
std::cout << tm2[0] << std::endl;
*/
// TODO: TRI
	//cv::Point3f* tm0 = (cv::Point3f*)trifocal_map_0.data;
	//v::Point3f* tm1 = (cv::Point3f*)trifocal_map_1.data;
	//cv::Point3f* tm2 = (cv::Point3f*)trifocal_map_2.data;
	// check if able to have at least one pose
	//if (n_points < 4)
	//{
	//	return 0;
	//}

//trifocal_rng_initialize(1, seeds);
		//int valid = trifocal_R_t_batch(cfg.n_poses_to_sample, 12, p2d_1, p2d_2, p2d_3, sp2d, (float*)&pts3[0], n_points, nullptr, rt1, rt2, nullptr, nullptr);
		//std::cout << "TRI BATCH VALID: " << valid << std::endl;
		//poses_pool_used = trifocal_R_t_batch(cfg.n_poses_to_sample, 12, p2d_1, p2d_2, p2d_3, sp2d, (float*)&pts3[0], n_points, nullptr, (float*)poses_pool.data, rt2, nullptr, nullptr);

		/*
		{
			float* base_r1 = r1 + (3 * sample_point);
			float* base_t1 = t1 + (3 * sample_point);

			std::cout << "FINAL result " << base_r1[0] << ","
				<< base_r1[1] << ","
				<< base_r1[2] << ","
				<< base_t1[0] << ","
				<< base_t1[1] << ","
				<< base_t1[2] << std::endl;
		}
		*/
		/*
						if (i == sample_point)
						{
							std::cout << points_2D_1[(2 * p) + 0] << ","
								<< points_2D_1[(2 * p) + 1] << ","
								<< points_2D_2[(2 * p) + 0] << ","
								<< points_2D_2[(2 * p) + 1] << ","
								<< points_2D_3[(2 * p) + 0] << ","
								<< points_2D_3[(2 * p) + 1] << ",SP,"
								<< base_2D[(2 * p) + 0] << ","
								<< base_2D[(2 * p) + 1] << ","
								<< points_3D[(3 * p) + 0] << ","
								<< points_3D[(3 * p) + 1] << ","
								<< points_3D[(3 * p) + 2] << ";" << std::endl;
						}
						*/
						/*
						if (i == sample_point)
						{
							std::cout << "result " << rt01[0] << ","
												   << rt01[1] << ","
												   << rt01[2] << ","
												   << rt01[3] << ","
												   << rt01[4] << ","
												   << rt01[5] << std::endl;
						}
						*/
						/*
								int seeds[16] = {
									982570992,
									156771448,
									1453167113,
									828969480,
									1428633992,
									1793003870,
									1450716004,
									583011327,
									225626512,
									770619902,
									1412352324,
									789718351,
									1363665346,
									203010848,
									604667542,
									1381341612,
								};
								*/

								//cv::Point2f const* pts2 = pts2_map.data();
									//cv::Point3f const* pts3 = pts3_map.data();
										//int n_points = (int)pts2_map.size();

										//for (int i = 0; i < cfg.n_poses_to_sample; i++) {
										//	int i1 = ((float)rand() / (float)RAND_MAX) * n_points;
										//	int i2 = ((float)rand() / (float)RAND_MAX) * n_points;
										//	int i3 = ((float)rand() / (float)RAND_MAX) * n_points;
										//	int i4 = ((float)rand() / (float)RAND_MAX) * n_points;


										//	if (cfg.lambdatwist) {
												// batch solve lambdatwist
											//}
											//else {
												// batch solve ap3p
											//}
										//}
/*
		int n_points = (int)pts2_map.size();

		float* ret_Rs = new float[cfg.n_poses_to_sample * 3];
		float* ret_ts = new float[cfg.n_poses_to_sample * 3];

		if (cfg.lambdatwist) {
			//solve_batch_p3p_lambdatwist_gpu((float*)pts3, (float*)pts2, ret_Rs, ret_ts, (float*)cams[active_idx].K.data, n_points, cfg.n_poses_to_sample);
		}
		else {
			//solve_batch_p3p_ap3p_gpu((float*)pts3, (float*)pts2, ret_Rs, ret_ts, (float*)cams[active_idx].K.data, n_points, cfg.n_poses_to_sample);
		}

		for (int i = 0; i < cfg.n_poses_to_sample; i++) {
			if (isfinite(ret_Rs[i * 3 + 0] + ret_Rs[i * 3 + 1] + ret_Rs[i * 3 + 2] + ret_ts[i * 3 + 0] + ret_ts[i * 3 + 1] + ret_ts[i * 3 + 2])) {
				// This is a solved bug, atan2 is more accurate than acos2 in rodrigues implementation
				// Due to GPU accuracy issue, GPU p3p sometimes gives pure zero rotation, this will cause incorrect covar in rg_refine process
				//if (rg_refine && (ret_Rs[i * 3 + 0] == 0 && ret_Rs[i * 3 + 1] == 0 && ret_Rs[i * 3 + 2] == 0))
					//continue;
				poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(ret_Rs[i * 3 + 0], ret_Rs[i * 3 + 1], ret_Rs[i * 3 + 2]);
				poses_pool.at<cv::Vec3f>(poses_pool_used++, 1) = cv::Vec3f(ret_ts[i * 3 + 0], ret_ts[i * 3 + 1], ret_ts[i * 3 + 2]);
			}
		}

		delete[] ret_Rs;
		delete[] ret_ts;
		*/

		/*
			//if (cfg.cpu_p3p)
			if (cfg.solver_select == 0)
			{

			}



			else if (1)
			{


				auto time_stamp = std::chrono::high_resolution_clock::now();



				std::cout << "TFT compute time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
			}
			else if (0)
			{
				std::cout << "TRI CYCLE ---------------------------------------" << std::endl;

				cv::Point3f const* tm0 = trifocal_map_0.data();
				cv::Point3f const* tm1 = trifocal_map_1.data();
				cv::Point3f const* tm2 = trifocal_map_2.data();

				int n_points = trifocal_map_0.size();

				float points_2D_1[2 * 7];
				float points_2D_2[2 * 7];
				float points_2D_3[2 * 7];
				float points_3D[3 * 7];
				float base_2D[2 * 7];
				float tft[27];
				float rt01[6];
				float rt02[6];
				float s01[1];
				float s02[1];

				auto time_stamp = std::chrono::high_resolution_clock::now();
				//int sample_point = 8191;//3458;

				for (int i = 0; i < cfg.n_poses_to_sample; i++)
				{
					for (int p = 0; p < 7; ++p)
					{
						int idx = ((float)rand() / (float)RAND_MAX) * n_points;
						//if (i == sample_point) { std::cout << "set " << i << " pick: " << idx << std::endl; }

						points_2D_1[(2 * p) + 0] = (tm0[idx].x - cfg.cx) / cfg.fx;
						points_2D_2[(2 * p) + 0] = (tm1[idx].x - cfg.cx) / cfg.fx;
						points_2D_3[(2 * p) + 0] = (tm2[idx].x - cfg.cx) / cfg.fx;
						base_2D[(2 * p) + 0] = (pts2[idx].x - cfg.cx) / cfg.fx;

						points_2D_1[(2 * p) + 1] = (tm0[idx].y - cfg.cy) / cfg.fy;
						points_2D_2[(2 * p) + 1] = (tm1[idx].y - cfg.cy) / cfg.fy;
						points_2D_3[(2 * p) + 1] = (tm2[idx].y - cfg.cy) / cfg.fy;
						base_2D[(2 * p) + 1] = (pts2[idx].y - cfg.cy) / cfg.fy;

						memcpy(points_3D + (3 * p) + 0, &pts3[idx], sizeof(cv::Point3f));
					}

					trifocal_R_t(points_2D_1, points_2D_2, points_2D_3, base_2D, points_3D, tft, rt01, rt02, s01, s02);

					if (isfinite(rt01[0] + rt01[1] + rt01[2] + rt01[3] + rt01[4] + rt01[5]))
					{
						poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(rt01[0], rt01[1], rt01[2]);
						poses_pool.at<cv::Vec3f>(poses_pool_used++, 1) = cv::Vec3f(rt01[3], rt01[4], rt01[5]);
					}
				}
				std::cout << "TRI POOLS: " << poses_pool_used << std::endl;
				std::cout << "TFT compute time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
			}
			else if (0)
			{
				std::cout << "TRI BATCH CYCLE ---------------------------------------" << std::endl;
				auto time_stamp = std::chrono::high_resolution_clock::now();

				cv::Point3f const* tm0 = trifocal_map_0.data();
				cv::Point3f const* tm1 = trifocal_map_1.data();
				cv::Point3f const* tm2 = trifocal_map_2.data();

				int n_points = trifocal_map_0.size();

				float* p2d_1 = new float[n_points * 2];
				float* p2d_2 = new float[n_points * 2];
				float* p2d_3 = new float[n_points * 2];
				float* sp2d = new float[n_points * 2];
				float* sp3d = new float[n_points * 3];
				float* rt1 = new float[cfg.n_poses_to_sample * 6];
				float* rt2 = new float[cfg.n_poses_to_sample * 6];

				int sample_point = 8191;

				for (int i = 0; i < n_points; ++i)
				{
					p2d_1[(i * 2) + 0] = (tm0[i].x - cfg.cx) / cfg.fx;
					p2d_2[(i * 2) + 0] = (tm1[i].x - cfg.cx) / cfg.fx;
					p2d_3[(i * 2) + 0] = (tm2[i].x - cfg.cx) / cfg.fx;
					sp2d[(i * 2) + 0] = (pts2[i].x - cfg.cx) / cfg.fx;

					p2d_1[(i * 2) + 1] = (tm0[i].y - cfg.cy) / cfg.fy;
					p2d_2[(i * 2) + 1] = (tm1[i].y - cfg.cy) / cfg.fy;
					p2d_3[(i * 2) + 1] = (tm2[i].y - cfg.cy) / cfg.fy;
					sp2d[(i * 2) + 1] = (pts2[i].y - cfg.cy) / cfg.fy;
				}

				int valid = trifocal_R_t_batch(cfg.n_poses_to_sample, 12, p2d_1, p2d_2, p2d_3, sp2d, (float*)&pts3[0], n_points, nullptr, rt1, rt2, nullptr, nullptr);

				poses_pool_used = 0;
				for (int i = 0; i < valid; ++i)
				{
					poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(rt1[(6 * i) + 0], rt1[(6 * i) + 1], rt1[(6 * i) + 2]);
					poses_pool.at<cv::Vec3f>(poses_pool_used++, 1) = cv::Vec3f(rt1[(6 * i) + 3], rt1[(6 * i) + 4], rt1[(6 * i) + 5]);
				}

				std::cout << "TRI BATCH POOLS: " << poses_pool_used << std::endl;

				delete[] p2d_1;
				delete[] p2d_2;
				delete[] p2d_3;
				delete[] sp2d;
				delete[] sp3d;
				delete[] rt1;
				delete[] rt2;

				std::cout << "TFT compute time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
			}
			else
			{
				std::cout << "TRIKK CYCLE ---------------------------------------" << std::endl;

				int n_points = trifocal_map_0.size();

				cv::Point3f const* tm0 = trifocal_map_0.data();
				cv::Point3f const* tm1 = trifocal_map_1.data();
				cv::Point3f const* tm2 = trifocal_map_2.data();

				float points_2D_1[2 * 7];
				float points_2D_2[2 * 7];
				float points_2D_3[2 * 7];
				float points_3D[3 * 7];
				float base_2D[2 * 7];

				float r1[3];
				float r2[3];
				float t1[3];
				float t2[3];

				//float width = trifocal_map_0.cols;
				//float height = trifocal_map_0.rows;

				auto time_stamp = std::chrono::high_resolution_clock::now();

				for (int i = 0; i < cfg.n_poses_to_sample; i++)
				{
					for (int p = 0; p < 5; ++p)
					{
						int idx = ((float)rand() / (float)RAND_MAX) * n_points;

						points_2D_1[(2 * p) + 0] = tm0[idx].x;
						points_2D_2[(2 * p) + 0] = tm1[idx].x;
						points_2D_3[(2 * p) + 0] = tm2[idx].x;
						base_2D[(2 * p) + 0] = (pts2[idx].x - cfg.cx) / cfg.fx;

						points_2D_1[(2 * p) + 1] = tm0[idx].y;
						points_2D_2[(2 * p) + 1] = tm1[idx].y;
						points_2D_3[(2 * p) + 1] = tm2[idx].y;
						base_2D[(2 * p) + 1] = (pts2[idx].y - cfg.cy) / cfg.fy;

						memcpy(points_3D + (3 * p) + 0, &pts3[idx], sizeof(cv::Point3f));
					}

					//trifocal_R_t_poselib(points_2D_1, points_2D_2, points_2D_3, base_2D, points_3D, cfg.fx, cfg.fy, cfg.cx, cfg.cy, width, height, 5, r1, t1, r2, t2);

					std::cout << "r1 " << r1[0] << "," << r1[1] << "," << r1[2] << std::endl;

					if (isfinite(r1[0] + r1[1] + r1[2] + t1[0] + t1[1] + t1[2]))
					{
						poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(r1[0], r1[1], r1[2]);
						poses_pool.at<cv::Vec3f>(poses_pool_used++, 1) = cv::Vec3f(t1[0], t1[1], t1[2]);
					}
				}
				std::cout << "TRIKK POOLS: " << poses_pool_used << std::endl;
				std::cout << "TRIKK compute time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << std::endl;
			}
			*/