
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include <lambdatwist/lambdatwist_p4p.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_p4p_lambdatwist(float* p3d_1, float* p2d_2, float* r_12, float* t_12)
{
	float R_s[3][3];

	cv::Vec3f r_s;
	cv::Vec3f t_s;

	bool ok = lambdatwist_p4p<double, float, 5>(&p2d_2[0], &p2d_2[1], &p2d_2[2], &p2d_2[3], &p3d_1[0], &p3d_1[1], &p3d_1[2], &p3d_1[3], 1, 1, 0, 0, R_s, t_s.val);
	if (!ok) { return false; }

	cv::Rodrigues(cv::Matx33f(reinterpret_cast<float*>(R_s)), r_s);

	Eigen::Matrix<float, 3, 1> r = matrix_from_buffer<float, 3, 1>(r_s.val);
	Eigen::Matrix<float, 3, 1> t = matrix_from_buffer<float, 3, 1>(t_s.val);

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
