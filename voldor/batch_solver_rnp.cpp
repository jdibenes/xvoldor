
#include <opencv2/calib3d.hpp>
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_arguments
{
	cv::Point2f const* p2d;
	cv::Point3f const* p3d;
	float fx;
	float fy;
	float cx;
	float cy;
	int solver;
	bool direction;
	float r0;
	int max_pow;
	int max_iterations;
};

static void block_cpu_solver_rnp(job_descriptor& jd)
{
	job_arguments* ja = static_cast<job_arguments*>(jd.inputs);

	float* out = &((float*)jd.output)[jd.start * 6];

	for (int i = jd.start; i < jd.end; ++i)
	{
		cv::Point2f p2d[7];
		cv::Point3f p3d[7];

		for (int j = 0; j < jd.sample_size; ++j)
		{
			int s = jd.rng[(i * jd.sample_size) + j];

			p2d[j] = ja->p2d[s];
			p3d[j] = ja->p3d[s];

			p2d[j].x = (p2d[j].x - ja->cx) / ja->fx;
			p2d[j].y = (p2d[j].y - ja->cy) / ja->fy;
		}

		float r[3];
		float t[3];

		bool ok;

		switch (ja->solver)
		{
		case 1:  ok = solver_r6p2l(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t);                     break;
		case 2:  ok = solver_r6p2i(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t, ja->max_iterations); break;
		case 0:  
		default: ok = solver_r6p1l(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t);                     break;
		}

		if (!ok) { continue; }

		int index = jd.valid++;

		out[6 * index + 0] = r[0];
		out[6 * index + 1] = r[1];
		out[6 * index + 2] = r[2];
		out[6 * index + 3] = t[0];
		out[6 * index + 4] = t[1];
		out[6 * index + 5] = t[2];
	}
}

int batch_cpu_solver_rnp(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, int sample_size, cv::Mat const& K, int solver, bool direction, float r0, int max_pow, int max_iterations, int poses_to_sample, float* poses, int workers)
{
	job_arguments ja;

	ja.p2d = p2d;
	ja.p3d = p3d;
	ja.fx = K.at<float>(0, 0);
	ja.fy = K.at<float>(1, 1);
	ja.cx = K.at<float>(0, 2);
	ja.cy = K.at<float>(1, 2);
	ja.solver = solver;
	ja.direction = direction;
	ja.r0 = r0;
	ja.max_pow = max_pow;
	ja.max_iterations = max_iterations;

	return batch_solve(poses_to_sample, workers, block_cpu_solver_rnp, &ja, point_count, sample_size, false, poses, 6);
}
