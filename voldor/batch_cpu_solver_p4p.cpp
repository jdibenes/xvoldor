
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

static void batch_cpu_solver_p4p(job_descriptor& jd)
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

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_p4p, &ji, point_count, sample_size, unique, poses);
	return batch_finalize(jr, poses, 6);
}
