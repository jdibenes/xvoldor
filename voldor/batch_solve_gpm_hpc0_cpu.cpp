
#include <opencv2/calib3d.hpp>
#include "solver_gpm_hpc0.h"
#include <iostream>

// points in format [u, v, z]
int batch_solve_gpm_hpc0_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
   	int n_points = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int i1 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i2 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

		cv::Point3f pa1 = pts0[i1];
		cv::Point3f pb1 = pts0[i2];
		cv::Point3f pa2 = pts1[i1];
		cv::Point3f pb2 = pts1[i2];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;

		solver_gpm_hpc0((float*)&pa1, (float*)&pb1, (float*)&pa2, (float*)&pb2, (float*)&r, (float*)&t);

		float t_sum = t[0] + t[1] + t[2];
		float r_sum = r[0] + r[1] + r[2];

		float x_sum = t_sum + r_sum;

		if (!isfinite(x_sum)) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;
		poses_pool_used++;
	}

	return poses_pool_used;
}
