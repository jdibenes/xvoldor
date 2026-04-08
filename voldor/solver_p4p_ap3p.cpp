
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_p4p_ap3p(float* p3d_1, float* p2d_2, float* r_12, float* t_12)
{
	cv::Vec3d r_s;
	cv::Vec3d t_s;

	bool ok = solvePnP(cv::_InputArray(reinterpret_cast<cv::Point3f*>(p3d_1), 4), cv::_InputArray(reinterpret_cast<cv::Point2f*>(p2d_2), 4), cv::Mat::eye(3, 3, CV_32F), cv::Mat(), r_s, t_s, false, cv::SOLVEPNP_AP3P);
	if (!ok) { return false; }

	Eigen::Matrix<double, 3, 1> r = matrix_from_buffer<double, 3, 1>(r_s.val);
	Eigen::Matrix<double, 3, 1> t = matrix_from_buffer<double, 3, 1>(t_s.val);

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
