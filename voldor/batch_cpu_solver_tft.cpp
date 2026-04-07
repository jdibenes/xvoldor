
#include <opencv2/calib3d.hpp>
#include "batch_cpu_solver.h"
#include "solvers.h"

struct job_inputs
{
	cv::Point3f const* p3d_1;
	cv::Point2f const* p2d_2;
	cv::Point2f const* p2d_3;
	float fx;
	float fy;
	float cx;
	float cy;
	float* next_pool;
};

struct job_output
{
	float* poses;
	float* next_pool;
};

static void batch_cpu_solver_tft(job_descriptor& jd)
{
	job_inputs* ja = static_cast<job_inputs*>(jd.inputs);
	job_output* jo = static_cast<job_output*>(jd.output);

	cv::Point3f p1[7];
	cv::Point2f p2[7];
	cv::Point2f p3[7];

	float r1[3];
	float t1[3];
	float r2[3];
	float t2[3];

	for (int i = jd.start; i < jd.end; ++i)
	{
		int const* p = get_sample_indices(jd, i);

		for (int m = 0; m < jd.sample_size; ++m)
		{
			p1[m] = ja->p3d_1[p[m]];
			p2[m] = ja->p2d_2[p[m]];
			p3[m] = ja->p2d_3[p[m]];

			p2[m].x = (p2[m].x - ja->cx) / ja->fx;
			p2[m].y = (p2[m].y - ja->cy) / ja->fy;

			p3[m].x = (p3[m].x - ja->cx) / ja->fx;
			p3[m].y = (p3[m].y - ja->cy) / ja->fy;
		}

		bool ok = solver_tft_linear((float*)p1, (float*)p2, (float*)p3, jd.sample_size, r1, t1, r2, t2);

		if (!ok) { continue; }
		if (!is_valid_solution_6(r1, t1) || !is_valid_solution_6(r2, t2)) { continue; }

		put_solution_6(jd, jo->poses, r1, t1);
		put_solution_6(jd, jo->next_pool, r2, t2);

		jd.valid++;
	}
}

int batch_cpu_solver_tft(cv::Point3f const* p3d_1, cv::Point2f const* p2d_2, cv::Point2f const* p2d_3, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, float* next_pool, int workers, bool unique)
{
	job_inputs ji;
	job_output jo;

	ji.p3d_1 = p3d_1;
	ji.p2d_2 = p2d_2;
	ji.p2d_3 = p2d_3;
	ji.fx = K.at<float>(0, 0);
	ji.fy = K.at<float>(1, 1);
	ji.cx = K.at<float>(0, 2);
	ji.cy = K.at<float>(1, 2);

	jo.poses = poses;
	jo.next_pool = next_pool;

	std::vector<job_result> jr = batch_solve(poses_to_sample, workers, batch_cpu_solver_tft, &ji, point_count, 7, unique, &jo);
	batch_finalize(jr, poses, 6);
	return batch_finalize(jr, next_pool, 6);
}










// points in format [u, v, z]
//int batch_solve_tft_linear_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, std::vector<cv::Point3f> const& pts2, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, std::vector<cv::Vec6f>* next_pool)
//{
	/*
	int n_points = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r1;
	cv::Vec3f t1;
	cv::Vec6f rt2;

	Eigen::Matrix<float, 2, 7> p2d_0;
	Eigen::Matrix<float, 2, 7> p2d_1;
	Eigen::Matrix<float, 2, 7> p2d_2;
	Eigen::Matrix<float, 3, 7> p3d_0;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[7];

		sample(n_points, 7, ix);

		for (int j = 0; j < 7; ++j)
		{
			int m = ix[j]; //(int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

			cv::Point3f p0 = pts0[m];
			cv::Point3f p1 = pts1[m];
			cv::Point3f p2 = pts2[m];

			p2d_0(0, j) = (p0.x - cx) / fx;
			p2d_0(1, j) = (p0.y - cy) / fy;

			p2d_1(0, j) = (p1.x - cx) / fx;
			p2d_1(1, j) = (p1.y - cy) / fy;

			p2d_2(0, j) = (p2.x - cx) / fx;
			p2d_2(1, j) = (p2.y - cy) / fy;

			p3d_0(0, j) = p2d_0(0, j) * p0.z;
			p3d_0(1, j) = p2d_0(1, j) * p0.z;
			p3d_0(2, j) = p0.z;
		}

		bool ok = trifocal_R_t_linear(p2d_0.data(), p2d_1.data(), p2d_2.data(), p3d_0.data(), 7, true, (float*)&r1, (float*)&t1, (float*)&rt2, ((float*)&rt2) + 3);
		if (!ok) { continue; }

		if (((r1[0] == 0.0f) && (r1[1] == 0.0f) && (r1[2] == 0.0f)) || ((rt2[0] == 0.0f) && (rt2[1] == 0.0f) && (rt2[2] == 0.0f))) {
			continue;
		}

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r1;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t1;

		poses_pool_used++;

		if (next_pool) { next_pool->push_back(rt2); }
	}

	return poses_pool_used;
	*/
	//	return 0;
//}
