
#include <limits>
#include <opencv2/calib3d.hpp>
#include "../rolling_shutter/rnp.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p1l(float const* p3d, float const* p2d, bool direction, float r0, int maxpow, float* r01, float* t01)
{
	RSSinglelinCameraPoseVector solutions;

	Eigen::MatrixXd X7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d, 3, 7).cast<double>();
	Eigen::MatrixXd u7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d, 2, 7).cast<double>();
	
	Eigen::MatrixXd X = X7(Eigen::all, Eigen::seqN(0, 6));
	Eigen::MatrixXd u = u7(Eigen::all, Eigen::seqN(0, 6));

	R6P1Lin(X, u, direction, r0, maxpow, &solutions); // always returns 0

	if (solutions.size() < 0) { return false; }

	double max_error = std::numeric_limits<double>::infinity();

	Eigen::Matrix<double, 3, 1> rd;
	Eigen::Matrix<double, 3, 1> td;

	for (auto const& solution : solutions)
	{
	Eigen::Matrix<double, 3, 3> Rv = matrix_R_rodrigues(solution.v);
	Eigen::Matrix<double, 3, 1> tc = solution.C;

	double error = (u7.col(6) - ((Rv * X7.col(6)) + tc).colwise().hnormalized()).norm();
	if (error >= max_error) { continue; }
	max_error = error;

	rd = solution.v;
	td = tc;
	}

	Eigen::Matrix<float, 3, 1> r = rd.cast<float>();
	Eigen::Matrix<float, 3, 1> t = td.cast<float>();

	matrix_to_buffer(r, r01);
	matrix_to_buffer(t, t01);

	return true;
}

// OK
bool solver_r6p2l(float const* p3d, float const* p2d, bool direction, float r0, float* r01, float* t01)
{
	cv::Mat r_initial;
	cv::Mat t_initial;

	cv::solvePnP(cv::Mat(4, 3, CV_32FC1, (void*)p3d), cv::Mat(4, 2, CV_32FC1, (void*)p2d), cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), r_initial, t_initial, false, cv::SOLVEPNP_AP3P);

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>((double*)r_initial.data));

	Eigen::MatrixXd X7 = R_initial * matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d, 3, 7).cast<double>();
	Eigen::MatrixXd u7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d, 2, 7).cast<double>();

	Eigen::MatrixXd X = X7(Eigen::all, Eigen::seqN(0, 6));
	Eigen::MatrixXd u = u7(Eigen::all, Eigen::seqN(0, 6));

	RSDoublelinCameraPoseVector solutions;

	R6P2Lin(X, u, direction, r0, &solutions); // always returns 0

	if (solutions.size() < 0) { return false; }

	double max_error = std::numeric_limits<double>::infinity();

	Eigen::Matrix<double, 3, 3> Rd;
	Eigen::Matrix<double, 3, 1> td;

	for (auto const& solution : solutions)
	{
	Eigen::Matrix<double, 3, 3> Rv = Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v);
	Eigen::Matrix<double, 3, 1> tc = solution.C;

	double error = (u7.col(6) - ((Rv * X7.col(6)) + tc).colwise().hnormalized()).norm();
	if (error >= max_error) { continue; }
	max_error = error;

	Rd = Rv;
	td = tc;
	}

	Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues(Rd * R_initial).cast<float>();
	Eigen::Matrix<float, 3, 1> t = td.cast<float>();

	matrix_to_buffer(r, r01);
	matrix_to_buffer(t, t01);

	return true;
}

// OK
bool solver_r6pi(float const* p3d, float const* p2d, bool direction, float r0, int max_iterations, float* r01, float* t01)
{
	cv::Mat r_initial;
	cv::Mat t_initial;

	cv::solvePnP(cv::Mat(4, 3, CV_32FC1, (void*)p3d), cv::Mat(4, 2, CV_32FC1, (void*)p2d), cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), r_initial, t_initial, false, cv::SOLVEPNP_AP3P);

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>((double*)r_initial.data));

	Eigen::MatrixXd X7 = R_initial * matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d, 3, 7).cast<double>();
	Eigen::MatrixXd u7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d, 2, 7).cast<double>();

	Eigen::MatrixXd X = X7(Eigen::all, Eigen::seqN(0, 6));
	Eigen::MatrixXd u = u7(Eigen::all, Eigen::seqN(0, 6));

	RSDoublelinCameraPose solution;

	bool ok = !iterativeRnP<RSDoublelinCameraPose, R6PIter>(X, u, Eigen::Vector3d{0,0,0}, 6, r0, direction, max_iterations, solution);
	if (!ok) { return false; }

	Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues((Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v)) * R_initial).cast<float>();
	Eigen::Matrix<float, 3, 1> t = solution.C.cast<float>();

	matrix_to_buffer(r, r01);
	matrix_to_buffer(t, t01);

	return true;
}
