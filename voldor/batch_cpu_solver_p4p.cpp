
#include <iostream>
#include "helpers_opencv.h"
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_inputs
{
	cv::Point3f const* p3d_1;
	cv::Point2f const* p2k_2;
	float fx;
	float fy;
	float cx;
	float cy;
	int solver;
};

static void batch_cpu_solver_p4p_lambdatwist(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);

	cv::Point3f p3d_1[4];
	cv::Point2f p2d_2[4];

	float r[3];
	float t[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
	int const* p = get_sample_indices(jd, i);

	for (int m = 0; m < jd.sample_size; ++m)
	{
	int im = p[m];

	p3d_1[m] = ji->p3d_1[im];
	p2d_2[m] = p2k_to_p2d(ji->p2k_2[im], ji->fx, ji->fy, ji->cx, ji->cy);
	}

	bool ok;

	switch (ji->solver)
	{
	case 0:  ok = solver_p4p_ap3p(       reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), r, t); break;
	case 1:  ok = solver_p4p_lambdatwist(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), r, t); break;	
	default: ok = false;                                                                                          break;
	}

	if (!ok) { continue; }

	put_solution_6(jd, static_cast<float*>(jd.output), r, t);

	jd.valid++;
	}
}

int batch_cpu_solver_p4p(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, int workers, bool unique)
{
	job_inputs ji;

	ji.p3d_1 = p3d_1;
	ji.p2k_2 = p2k_2;	
	ji.fx = K.at<float>(0, 0);
	ji.fy = K.at<float>(1, 1);
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);
	ji.solver = solver;

	int sample_size;

	switch (solver)
	{
	case 0:  sample_size = 4; break;
	case 1:  sample_size = 4; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_p4p_lambdatwist, &ji, point_count, sample_size, unique, poses);



	int valid = batch_finalize(jr, poses, 6);
	std::cout << "VALID: " << valid << std::endl;
	return valid;
}







//float R[3][3];
//float t[3];
//cv::Vec3f r;
//int i1 = p[0];
		//int i2 = p[1];
		//int i3 = p[2];
		//int i4 = p[3];

		//bool ok = lambdatwist_p4p<double, float, 5>((float*)&ja->p2k[i1], (float*)&ja->p2k[i2], (float*)&ja->p2k[i3], (float*)&ja->p2k[i4], (float*)&ja->p3d[i1], (float*)&ja->p3d[i2], (float*)&ja->p3d[i3], (float*)&ja->p3d[i4], ja->fx, ja->fy, ja->cx, ja->cy, R, t);
		//if (!ok) { continue; }

		//Rodrigues(cv::Matx33f((float*)R), r);



		//if (!is_valid_solution_6(r.val, t)) { continue; }


/*
#include <opencv2/calib3d.hpp>

int batch_solve_ap3p_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
	int n_points = (int)pts2.size();
	int poses_pool_used = 0;

	cv::Vec3d rvec_temp;
	cv::Vec3d tvec_temp;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int i1 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i2 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i3 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i4 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

		cv::Point2f p2s_tmp[4] = { pts2[i1], pts2[i2], pts2[i3], pts2[i4] };
		cv::Point3f p3s_tmp[4] = { pts3[i1], pts3[i2], pts3[i3], pts3[i4] };

		//bool ok = solvePnP(cv::_InputArray(p3s_tmp, 4), cv::_InputArray(p2s_tmp, 4), K, cv::Mat(), rvec_temp, tvec_temp, false, cv::SOLVEPNP_AP3P);

		//if (!ok) { continue; }

		double t_sum = tvec_temp[0] + tvec_temp[1] + tvec_temp[2];
		double r_sum = rvec_temp[0] + rvec_temp[1] + rvec_temp[2];

		double x_sum = t_sum + r_sum;

		if (!isfinite(x_sum)) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = rvec_temp;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = tvec_temp;
		poses_pool_used++;
	}

	return poses_pool_used;
}
*/