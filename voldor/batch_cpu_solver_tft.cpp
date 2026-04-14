
#include "helpers_opencv.h"
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_inputs
{
	cv::Point3f const* p2z_1;
	cv::Point3f const* p2z_2;
	cv::Point3f const* p2z_3;
	float fx;
	float fy;
	float cx;
	float cy;
	int solver;
	float threshold;
};

struct job_output
{
	float* poses;
	float* next_pool;
};

static void batch_cpu_solver_tft(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);
	job_output* jo = static_cast<job_output*>(jd.output);

	cv::Point3f p3d_1[7];
	cv::Point2f p2d_2[7];
	cv::Point2f p2d_3[7];

	float r1[3];
	float t1[3];
	float r2[3];
	float t2[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
	int const* p = get_sample_indices(jd, i);

	for (int m = 0; m < jd.sample_size; ++m)
	{
	int im = p[m];

	p3d_1[m] = p2z_to_p3d(ji->p2z_1[im], ji->fx, ji->fy, ji->cx, ji->cy);
	p2d_2[m] = p2z_to_p2d(ji->p2z_2[im], ji->fx, ji->fy, ji->cx, ji->cy);
	p2d_3[m] = p2z_to_p2d(ji->p2z_3[im], ji->fx, ji->fy, ji->cx, ji->cy);
	}

	bool ok;
	
	switch (ji->solver)
	{
	case 0:  ok = solver_tft_linear(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), reinterpret_cast<float*>(p2d_3), jd.sample_size, r1, t1, r2, t2, ji->threshold); break;
	default: ok = false;                                                                                                                                                               break;
	}

	if (!ok) { continue; }

	put_solution_6(jd, jo->poses, r1, t1);
	if (jo->next_pool) { put_solution_6(jd, jo->next_pool, r2, t2); }

	jd.valid++;
	}
}

int batch_cpu_solver_tft(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2, cv::Point3f const* p2z_3, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* next_pool, int workers, bool unique, float threshold)
{
	job_inputs ji;
	job_output jo;

	ji.p2z_1 = p2z_1;
	ji.p2z_2 = p2z_2;
	ji.p2z_3 = p2z_3;
	ji.fx = K.at<float>(0, 0);
	ji.fy = K.at<float>(1, 1);
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);
	ji.solver = solver;
	ji.threshold = threshold;

	jo.poses = poses;
	jo.next_pool = next_pool;

	int sample_size;

	switch (solver)
	{
	case 0:  sample_size = 7; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_tft, &ji, point_count, sample_size, unique, &jo);
	if (next_pool) { batch_finalize(jr, next_pool, 6); }
	return batch_finalize(jr, poses, 6);
}
