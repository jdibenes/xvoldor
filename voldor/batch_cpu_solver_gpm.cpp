
#include "helpers_opencv.h"
#include "batch_cpu_solver.h"
#include "solvers.h"

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

static void batch_cpu_solver_gpm(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);

	cv::Point3f p3d_1[7];
	cv::Point3f p3d_2[7];

	float r[3];
	float t[3];

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
	case 0:  ok = solver_gpm_hpc0(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 1:  ok = solver_gpm_hpc1(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 2:  ok = solver_gpm_hpc2(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 3:  ok = false;                                                                                   break; // TODO: HPC3
	case 4:  ok = solver_gpm_m4(  reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 5:  ok = solver_gpm_nm5( reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 6:  ok = solver_gpm_nm6( reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	case 7:  ok = solver_gpm_nm7( reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p3d_2), r, t); break;
	default: ok = false;                                                                                   break;
	}

	if (!ok) { continue; }
	if (!is_valid_solution_6(r, t)) { continue; }

	put_solution_6(jd, static_cast<float*>(jd.output), r, t);

	jd.valid++;
	}
}

int batch_cpu_solver_gpm(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, int workers, bool unique)
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
	case 0:  sample_size = 2; break;
	case 1:  sample_size = 2; break;
	case 2:  sample_size = 3; break;
	case 3:  sample_size = 3; break;
	case 4:  sample_size = 4; break;
	case 5:  sample_size = 5; break;
	case 6:  sample_size = 6; break;
	case 7:  sample_size = 7; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_gpm, &ji, point_count, sample_size, unique, poses);
	return batch_finalize(jr, poses, 6);
}
