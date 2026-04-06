
#include <opencv2/calib3d.hpp>
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_inputs
{
	cv::Point3f const* p3d_1;
	cv::Point3f const* p3d_2;
	int solver;
};

static void batch_cpu_solver_gpm(job_descriptor& jd)
{
	job_inputs* ja = static_cast<job_inputs*>(jd.inputs);

	cv::Point3f p1[7];
	cv::Point3f p2[7];

	float r[3];
	float t[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
		int const* p = get_sample_indices(jd, i);

		for (int m = 0; m < jd.sample_size; ++m)
		{
			p1[m] = ja->p3d_1[p[m]];
			p2[m] = ja->p3d_2[p[m]];
		}

		bool ok;

		switch (ja->solver)
		{
		case 0:  ok = solver_gpm_hpc0((float*)p1, (float*)p2, r, t); break;
		case 1:  ok = solver_gpm_hpc1((float*)p1, (float*)p2, r, t); break;
		case 2:  ok = solver_gpm_hpc2((float*)p1, (float*)p2, r, t); break;
		case 3:  ok = false;                                         break; // TODO: HPC3
		case 4:  ok = solver_gpm_m4(  (float*)p1, (float*)p2, r, t); break;
		case 5:  ok = solver_gpm_nm5( (float*)p1, (float*)p2, r, t); break;
		case 6:  ok = solver_gpm_nm6( (float*)p1, (float*)p2, r, t); break;
		case 7:  ok = solver_gpm_nm7( (float*)p1, (float*)p2, r, t); break;
		default: ok = false;
		}

		if (!ok) { continue; }

		if (is_valid_solution_6(r, t)) { put_solution_6(jd, r, t); }
	}
}

int batch_cpu_solver_gpm(cv::Point3f const* p3d_1, cv::Point3f const* p3d_2, int point_count, int solver, int poses_to_sample, float* poses, int workers, bool unique)
{
	job_inputs ja;

	ja.p3d_1 = p3d_1;
	ja.p3d_2 = p3d_2;
	ja.solver = solver;

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

	return batch_solve(poses_to_sample, workers, batch_cpu_solver_gpm, &ja, point_count, sample_size, unique, poses, 6);
}







// points in format [u, v, z]
//int batch_cpu_solver_gpm_hpc0(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
//{




	/*
	int n_points        = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[2];

		sample(n_points, 2, ix);

		int i1 = ix[0];
		int i2 = ix[1];

		cv::Point3f p1[2];
		cv::Point3f p2[2];

		p1[0] = pts0[i1];
		p1[1] = pts0[i2];
		p2[0] = pts1[i1];
		p2[1] = pts1[i2];

		cv::Point3f& pa1 = p1[0];
		cv::Point3f& pb1 = p1[1];
		cv::Point3f& pa2 = p2[0];
		cv::Point3f& pb2 = p2[1];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;

		bool ok = solver_gpm_hpc0((float*)p1, (float*)p2, (float*)&r, (float*)&t);
		if (!ok) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;

		poses_pool_used++;
	}


	return poses_pool_used;
	*/


	/*
   	int n_points        = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[2];

		sample(n_points, 2, ix);

		int i1 = ix[0];
		int i2 = ix[1];

		cv::Point3f p1[2];
		cv::Point3f p2[2];

		p1[0] = pts0[i1];
		p1[1] = pts0[i2];
		p2[0] = pts1[i1];
		p2[1] = pts1[i2];

		cv::Point3f& pa1 = p1[0];
		cv::Point3f& pb1 = p1[1];
		cv::Point3f& pa2 = p2[0];
		cv::Point3f& pb2 = p2[1];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;

		bool ok = solver_gpm_hpc0((float*)p1, (float*)p2, (float*)&r, (float*)&t);
		if (!ok) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;

		poses_pool_used++;
	}
	

	return poses_pool_used;
	*/
	//return 0;
//}


//#include <opencv2/calib3d.hpp>
//#include "solvers.h"
//#include "batch_solve_common.h"

// points in format [u, v, z]
//int batch_solve_gpm_hpc1_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations)
//{
	/*
	int n_points        = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[2];

		sample(n_points, 2, ix);

		int i1 = ix[0];
		int i2 = ix[1];

		cv::Point3f p1[2];
		cv::Point3f p2[2];

		p1[0] = pts0[i1];
		p1[1] = pts0[i2];
		p2[0] = pts1[i1];
		p2[1] = pts1[i2];

		cv::Point3f& pa1 = p1[0];
		cv::Point3f& pb1 = p1[1];
		cv::Point3f& pa2 = p2[0];
		cv::Point3f& pb2 = p2[1];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;

		bool ok = solver_gpm_hpc1((float*)p1, (float*)p2, (float*)&r, (float*)&t);
		if (!ok) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;

		poses_pool_used++;
	}

	return poses_pool_used;
	*/
	//return 0;
//}

//#include <opencv2/calib3d.hpp>
//#include "solvers.h"
//#include "batch_solve_common.h"

// points in format [u, v, z]
//int batch_solve_gpm_hpc2_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations)
//{
	/*
	int n_points        = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[3];

		sample(n_points, 3, ix);

		int i1 = ix[0];
		int i2 = ix[1];
		int i3 = ix[2];

		cv::Point3f p1[3];
		cv::Point3f p2[3];

		p1[0] = pts0[i1];
		p1[1] = pts0[i2];
		p1[2] = pts0[i3];
		p2[0] = pts1[i1];
		p2[1] = pts1[i2];
		p2[2] = pts1[i3];

		cv::Point3f& pa1 = p1[0];
		cv::Point3f& pb1 = p1[1];
		cv::Point3f& pc1 = p1[2];
		cv::Point3f& pa2 = p2[0];
		cv::Point3f& pb2 = p2[1];
		cv::Point3f& pc2 = p2[2];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pc1.x = ((pc1.x - cx) / fx) * pc1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;
		pc2.x = ((pc2.x - cx) / fx) * pc2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pc1.y = ((pc1.y - cy) / fy) * pc1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;
		pc2.y = ((pc2.y - cy) / fy) * pc2.z;

		bool ok = solver_gpm_hpc2((float*)p1, (float*)p2, (float*)&r, (float*)&t);
		if (!ok) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;

		poses_pool_used++;
	}

	return poses_pool_used;
	*/
	//return 0;
//}
