
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
	bool direction;
	float r0;
	int max_iterations;
};

struct job_output
{
	float* poses;
	float* velocities;
};

static void block_cpu_solver_rnp(job_descriptor& jd)
{
	job_inputs* ji = static_cast<job_inputs*>(jd.inputs);
	job_output* jo = static_cast<job_output*>(jd.output);

	cv::Point3f p3d_1[7];
	cv::Point2f p2d_2[7];	

	float r[3];
	float t[3];

	float dr[3];
	float dt[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
	int const* p = get_sample_indices(jd, i);

	for (int m = 0; m < jd.sample_size; ++m)
	{
	int im = p[m];

	p3d_1[m] =            ji->p3d_1[im];
	p2d_2[m] = p2k_to_p2d(ji->p2k_2[im], ji->fx, ji->fy, ji->cx, ji->cy);
	}

	bool ok;

	switch (ji->solver)
	{
	case 0:  ok = solver_r6p1l(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), ji->direction, ji->r0, r, t, dr, dt);                     break;
	case 1:  ok = solver_r6p2l(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), ji->direction, ji->r0, r, t, dr, dt);                     break;
	case 2:  ok = solver_r6p2i(reinterpret_cast<float*>(p3d_1), reinterpret_cast<float*>(p2d_2), ji->direction, ji->r0, r, t, dr, dt, ji->max_iterations); break;
	default: ok = false;                                                                                                                                   break;
	}

	if (!ok) { continue; }

	put_solution_6(jd, jo->poses, r, t);
	if (jo->velocities) { put_solution_6(jd, jo->velocities, dr, dt); }

	jd.valid++;
	}
}

int batch_cpu_solver_rnp(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* velocities, int workers, bool unique, bool direction, float r0, int max_iterations)
{
	job_inputs ji;
	job_output jo;

	ji.p3d_1 = p3d_1;
	ji.p2k_2 = p2k_2;	
	ji.fx = K.at<float>(0, 0);
	ji.fy = K.at<float>(1, 1);
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);
	ji.solver = solver;
	ji.direction = direction;
	ji.r0 = r0;
	ji.max_iterations = max_iterations;

	jo.poses = poses;
	jo.velocities = velocities;

	int sample_size;

	switch (solver)
	{
	case 0:  sample_size = 7; break;
	case 1:  sample_size = 7; break;
	case 2:  sample_size = 6; break;
	default: return 0;
	}

	if (point_count < sample_size) { return 0; }

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, block_cpu_solver_rnp, &ji, point_count, sample_size, unique, &jo);
	if (velocities) { batch_finalize(jr, velocities, 6); }
	return batch_finalize(jr, poses, 6);
}
