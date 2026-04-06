
#include <opencv2/calib3d.hpp>
#include "batch_cpu_solver.h"
#include "helpers_geometry.h"
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

static void batch_cpu_solver_p4p_lambdatwist(job_descriptor& jd)
{
	job_inputs* ja = static_cast<job_inputs*>(jd.inputs);

	float R[3][3];
	float t[3];
	cv::Vec3f r;

	for (int i = jd.start; i < jd.end; ++i)
	{
		int const* p = get_sample_indices(jd, i);

		int i1 = p[0];
		int i2 = p[1];
		int i3 = p[2];
		int i4 = p[3];

		bool ok = lambdatwist_p4p<double, float, 5>((float*)&ja->p2d[i1], (float*)&ja->p2d[i2], (float*)&ja->p2d[i3], (float*)&ja->p2d[i4], (float*)&ja->p3d[i1], (float*)&ja->p3d[i2], (float*)&ja->p3d[i3], (float*)&ja->p3d[i4], ja->fx, ja->fy, ja->cx, ja->cy, R, t);
		if (!ok) { continue; }
		
		Rodrigues(cv::Matx33f((float*)R), r);
		
		if (is_valid_solution_6(r.val, t)) { put_solution_6(jd, r.val, t); }		
	}
}

int batch_cpu_solver_p4p_lambdatwist(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, int workers, bool unique)
{
	job_inputs ja;

	ja.p2d = p2d;
	ja.p3d = p3d;
	ja.fx = K.at<float>(0, 0);
	ja.fy = K.at<float>(1, 1);
	ja.cx = K.at<float>(0, 2);
	ja.cy = K.at<float>(1, 2);

	return batch_solve(poses_to_sample, workers, batch_cpu_solver_p4p_lambdatwist, &ja, point_count, 4, unique, poses, 6);
}
