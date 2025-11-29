
#include <opencv2/calib3d.hpp>
#include "../lambdatwist/lambdatwist_p4p.h"

int batch_solve_lambdatwist_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
	int n_points = (int)pts2.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	float R_temp[3][3];

	cv::Vec3f rvec_temp;
	cv::Vec3f tvec_temp;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int i1 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i2 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i3 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i4 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

		bool ok = lambdatwist_p4p<double, float, 5>((float*)&pts2[i1], (float*)&pts2[i2], (float*)&pts2[i3], (float*)&pts2[i4], (float*)&pts3[i1], (float*)&pts3[i2], (float*)&pts3[i3], (float*)&pts3[i4], fx, fy, cx, cy, R_temp, tvec_temp.val);

		if (!ok) { continue; }

		Rodrigues(cv::Matx33f((float*)R_temp), rvec_temp);

		float t_sum = tvec_temp[0] + tvec_temp[1] + tvec_temp[2];
		float r_sum = rvec_temp[0] + rvec_temp[1] + rvec_temp[2];

		float x_sum = t_sum + r_sum;

		if (!isfinite(x_sum)) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = rvec_temp;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = tvec_temp;
		poses_pool_used++;
	}

	return poses_pool_used;
}
