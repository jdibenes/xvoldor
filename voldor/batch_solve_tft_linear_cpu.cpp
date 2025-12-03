
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include "batch_solve_common.h"
#include "trifocal.h"

// points in format [u, v, z]
int batch_solve_tft_linear_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, std::vector<cv::Point3f> const& pts2, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, std::vector<cv::Vec6f>* next_pool)
{
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
}
