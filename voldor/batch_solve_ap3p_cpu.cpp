
#include <opencv2/calib3d.hpp>

int batch_solve_ap3p_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
	int n_points = (int)pts2.size();
	int poses_pool_used = 0;

	cv::Vec3d rvec_temp;
	cv::Vec3d tvec_temp;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int i1 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i2 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i3 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));
		int i4 = (int)(((float)rand() / (float)RAND_MAX) * (n_points - 1));

		cv::Point2f p2s_tmp[4] = { pts2[i1], pts2[i2], pts2[i3], pts2[i4] };
		cv::Point3f p3s_tmp[4] = { pts3[i1], pts3[i2], pts3[i3], pts3[i4] };

		bool ok = solvePnP(cv::_InputArray(p3s_tmp, 4), cv::_InputArray(p2s_tmp, 4), K, cv::Mat(), rvec_temp, tvec_temp, false, cv::SOLVEPNP_AP3P);
		
		if (!ok) { continue; }

		double t_sum = tvec_temp[0] + tvec_temp[1] + tvec_temp[2];
		double r_sum = rvec_temp[0] + rvec_temp[1] + rvec_temp[2];
		
		double x_sum = t_sum + r_sum;

		if (!isfinite(x_sum)) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = rvec_temp;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = tvec_temp;
		poses_pool_used++;
	}

	return poses_pool_used;
}
