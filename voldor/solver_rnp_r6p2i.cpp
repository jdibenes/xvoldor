
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p2i(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, int max_iterations)
{
	cv::Mat r_initial;
	cv::Mat t_initial;

	bool ig = cv::solvePnP(cv::Mat(4, 3, CV_32FC1, (void*)p3d_1), cv::Mat(4, 2, CV_32FC1, (void*)p2d_2), cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), r_initial, t_initial, false, cv::SOLVEPNP_AP3P);
	if (!ig) { return false; }

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>((double*)r_initial.data));

	Eigen::MatrixXd X6 = R_initial * matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 6).cast<double>();
	Eigen::MatrixXd u6 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, 6).cast<double>();

	RSDoublelinCameraPose solution;

	bool ok = !iterativeRnP<RSDoublelinCameraPose, R6PIter>(X6, u6, Eigen::Vector3d{ 0, 0, 0 }, 6, r0, direction, max_iterations, solution); // vk, sampleSize not used
	if (!ok) { return false; }

	Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues((Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v)) * R_initial).cast<float>();
	Eigen::Matrix<float, 3, 1> t = solution.C.cast<float>();

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
