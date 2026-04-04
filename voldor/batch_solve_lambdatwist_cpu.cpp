
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "batch_solve_cpu.h"
#include "../lambdatwist/lambdatwist_p4p.h"

struct job_inputs
{
	cv::Point2f const* p2d;
	cv::Point3f const* p3d;
	float fx;
	float fy;
	float cx;
	float cy;
};


static void block_cpu_solver_p4p_lambdatwist(job_descriptor& jd)
{
	job_inputs* ja = static_cast<job_inputs*>(jd.inputs);

	float R_temp[3][3];
	float t_temp[3];

	cv::Vec3f rvec_temp;
	cv::Vec3f tvec_temp;

	//std::cout << "JOB: " << jd.id << "(" << jd.start << "->" << jd.end << ") : " << jd.seed << std::endl;

	float* out = &((float*)jd.output)[jd.start * 6];

	for (int i = jd.start; i < jd.end; ++i)
	{
		int i1 = jd.rng[(i * jd.sample_size) + 0];
		int i2 = jd.rng[(i * jd.sample_size) + 1];
		int i3 = jd.rng[(i * jd.sample_size) + 2];
		int i4 = jd.rng[(i * jd.sample_size) + 3];

		bool ok = lambdatwist_p4p<double, float, 5>((float*)&ja->p2d[i1], (float*)&ja->p2d[i2], (float*)&ja->p2d[i3], (float*)&ja->p2d[i4],
			                                        (float*)&ja->p3d[i1], (float*)&ja->p3d[i2], (float*)&ja->p3d[i3], (float*)&ja->p3d[i4],
			                                        ja->fx, ja->fy, ja->cx, ja->cy, R_temp, tvec_temp.val);

		if (!ok) { continue; }

		Rodrigues(cv::Matx33f((float*)R_temp), rvec_temp);

		float t_sum = tvec_temp[0] + tvec_temp[1] + tvec_temp[2];
		float r_sum = rvec_temp[0] + rvec_temp[1] + rvec_temp[2];

		float x_sum = t_sum + r_sum;

		if (!std::isfinite(x_sum)) { continue; }
		
		out[6 * jd.valid + 0] = rvec_temp[0];
		out[6 * jd.valid + 1] = rvec_temp[1];
		out[6 * jd.valid + 2] = rvec_temp[2];
		out[6 * jd.valid + 3] = tvec_temp[0];
		out[6 * jd.valid + 4] = tvec_temp[1];
		out[6 * jd.valid + 5] = tvec_temp[2];

		jd.valid++;
	}
}

int batch_cpu_solver_p4p_lambdatwist(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, int workers)
{
	job_inputs ja;

	ja.p2d = p2d;
	ja.p3d = p3d;
	ja.fx = K.at<float>(0, 0);
	ja.fy = K.at<float>(1, 1);
	ja.cx = K.at<float>(0, 2);
	ja.cy = K.at<float>(1, 2);

	int valid =  batch_solve(poses_to_sample, workers, block_cpu_solver_p4p_lambdatwist, &ja, point_count, 4, poses, 6*sizeof(float));
	std::cout << "VALID: " << valid << std::endl;
	return valid;
}








/*
int batch_solve_lambdatwist_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
	int n_points = (int)pts2.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	float R_temp[3][3];

	cv::Vec3f rvec_temp;
	cv::Vec3f tvec_temp;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int i1 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i2 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i3 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i4 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

		bool ok = lambdatwist_p4p<double, float, 5>((float*)&pts2[i1], (float*)&pts2[i2], (float*)&pts2[i3], (float*)&pts2[i4], (float*)&pts3[i1], (float*)&pts3[i2], (float*)&pts3[i3], (float*)&pts3[i4], fx, fy, cx, cy, R_temp, tvec_temp.val);

		if (!ok) { continue; }

		Rodrigues(cv::Matx33f((float*)R_temp), rvec_temp);

		float t_sum = tvec_temp[0] + tvec_temp[1] + tvec_temp[2];
		float r_sum = rvec_temp[0] + rvec_temp[1] + rvec_temp[2];

		float x_sum = t_sum + r_sum;

		if (!isfinite(x_sum)) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = rvec_temp;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = tvec_temp;
		poses_pool_used++;
	}

	return poses_pool_used;
}
*/