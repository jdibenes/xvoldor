
#include "helpers_opencv.h"
#include "batch_cpu_solver.h"
#include "detect.h"

struct job_inputs
{
	cv::Point3f const* p2z_1;
	cv::Point3f const* p2z_2;
	float fx;
	float fy;
	float cx;
	float cy;
	int solver;
};

static void batch_cpu_detect_planar(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);

	cv::Point3f p3d_1[3];
	cv::Point3f p3d_2[3];

	float f;

	for (int i = jd.start; i < jd.end; ++i)
	{
	int const* p = get_sample_indices(jd, i);

	for (int m = 0; m < jd.sample_size; ++m)
	{
	int im = p[m];

	p3d_1[m] = p2z_to_p3d(ji->p2z_1[im], ji->fx, ji->fy, ji->cx, ji->cy);
	p3d_2[m] = p2z_to_p3d(ji->p2z_2[im], ji->fx, ji->fy, ji->cx, ji->cy);
	}

	bool ok;

	switch (ji->solver)
	{
	case 0:  ok = detect_planar_3d(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), &f, 1e-6, 1e-6); break;
	default: ok = false;                                                                                                           break;
	}

	if (!ok) { continue; }

	put_solution_1(jd, static_cast<float*>(jd.output), &f);

	jd.valid++;
	}
}

int batch_cpu_detect_planar(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, int workers, bool unique)
{
	job_inputs ji;

	ji.p2z_1 = p2z_1;
	ji.p2z_2 = p2z_2;
	ji.fx = K.at<float>(0, 0);
	ji.fy = K.at<float>(1, 1);
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);
	ji.solver = solver;

	int sample_size;

	switch (solver)
	{
	case 0:  sample_size = 3; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_detect_planar, &ji, point_count, sample_size, unique, poses);
	return batch_finalize(jr, poses, 1);
}
