
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p2i(float* p3d_1, float* p2d_2, bool direction, float r0, float* r_12, float* t_12, int max_iterations)
{
	cv::Mat r_initial;
	cv::Mat t_initial;

	bool ig = cv::solvePnP(cv::Mat(4, 3, CV_32FC1, p3d_1), cv::Mat(4, 2, CV_32FC1, p2d_2), cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), r_initial, t_initial, false, cv::SOLVEPNP_AP3P);
	if (!ig) { return false; }

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>(reinterpret_cast<double*>(r_initial.data)));

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X6 = R_initial * matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 6);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u6 =             matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, 6);

	RSDoublelinCameraPose solution;

	bool ok = !iterativeRnP<RSDoublelinCameraPose, R6PIter>(X6, u6, Eigen::Vector3d{ 0, 0, 0 }, 6, r0, direction, max_iterations, solution); // vk, sampleSize not used
	if (!ok) { return false; }

	Eigen::Matrix<double, 3, 1> r = vector_r_rodrigues((Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v)) * R_initial);
	Eigen::Matrix<double, 3, 1> t = solution.C;

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
