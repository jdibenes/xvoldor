
#include "helpers_opencv.h"
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_inputs
{
	cv::Point3f const* p3d_1;
	cv::Point2f const* p2k_2;
	float cx;
	float cy;
	int solver;
};

struct job_output
{
	float* poses;
	float* focals;
};

static void batch_cpu_solver_ppf(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);
	job_output* jo = static_cast<job_output*>(jd.output);

	cv::Point3f p3d_1[5];
	cv::Point2f p2k_2[5];

	float r[3];
	float t[3];
	float f[2];

	for (int i = jd.start; i < jd.end; ++i)
	{
	int const* p = get_sample_indices(jd, i);

	for (int m = 0; m < jd.sample_size; ++m)
	{
	int im = p[m];

	p3d_1[m] = ji->p3d_1[im];
	p2k_2[m] = ji->p2k_2[im];
	}

	bool ok;

	switch (ji->solver)
	{
	case 0:  ok = solver_ppf_p4pf(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2k_2), ji->cx, ji->cy, r, t, f); break;
	default: ok = false;                                                                                                      break;
	}

	if (!ok) { continue; }

	put_solution_6(jd, jo->poses, r, t);
	if (jo->focals) { put_solution_f(jd, jo->focals, f); }

	jd.valid++;
	}
}

int batch_cpu_solver_ppf(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* focals, int workers, bool unique)
{
	job_inputs ji;
	job_output jo;

	ji.p3d_1 = p3d_1;
	ji.p2k_2 = p2k_2;
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);
	ji.solver = solver;

	jo.poses = poses;
	jo.focals = focals;

	int sample_size;

	switch (solver)
	{
	case 0:  sample_size = 5; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_ppf, &ji, point_count, sample_size, unique, &jo);
	if (focals) { batch_finalize(jr, focals, 2); }
	return batch_finalize(jr, poses, 6);
}
