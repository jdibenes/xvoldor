
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

	cv::Point2f p2d[7];
	cv::Point3f p3d[7];

	float r[3];
	float t[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
		int const* p = get_sample_indices(jd, i);

		for (int m = 0; m < jd.sample_size; ++m)
		{
			p2d[m] = ja->p2d[p[m]];
			p3d[m] = ja->p3d[p[m]];

			p2d[m].x = (p2d[m].x - ja->cx) / ja->fx;
			p2d[m].y = (p2d[m].y - ja->cy) / ja->fy;
		}

		bool ok;

		switch (ja->solver)
		{
		case 1:  ok = solver_r6p2l(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t);                     break;
		case 2:  ok = solver_r6p2i(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t, ja->max_iterations); break;
		case 0:  
		default: ok = solver_r6p1l(reinterpret_cast<float*>(p3d), reinterpret_cast<float*>(p2d), ja->direction, ja->r0, r, t);                     break;
		}

		if (!ok) { continue; }
		if (!is_valid_solution_6(r, t)) { continue; }

		put_solution_6(jd, (float*)jd.output, r, t);

		jd.valid++;
	}
}

int batch_cpu_solver_rnp(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, int sample_size, cv::Mat const& K, int solver, bool direction, float r0, int max_pow, int max_iterations, int poses_to_sample, float* poses, int workers, bool unique)
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

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, block_cpu_solver_rnp, &ja, point_count, sample_size, unique, poses);
	return batch_finalize(jr, poses, 6);
}
