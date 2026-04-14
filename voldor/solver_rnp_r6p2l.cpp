
#include <limits>
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p2l(float const* p3d_1, float const* p2d_2, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12)
{
	cv::Mat r_initial;
	cv::Mat t_initial;

	bool ok = cv::solvePnP({ reinterpret_cast<cv::Point3f const*>(p3d_1), 4 }, { reinterpret_cast<cv::Point2f const*>(p2d_2), 4 }, cv::Mat::eye(3, 3, CV_32FC1), cv::Mat(), r_initial, t_initial, false, cv::SOLVEPNP_AP3P);
	if (!ok) { return false; }

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>(reinterpret_cast<double*>(r_initial.data)));

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X7 = R_initial * matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 7);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u7 =             matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, 7);

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X = X7(Eigen::indexing::all, Eigen::seqN(0, 6));
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u = u7(Eigen::indexing::all, Eigen::seqN(0, 6));

	RSDoublelinCameraPoseVector solutions;

	R6P2Lin(X, u, direction, r0, &solutions); // always returns 0

	if (solutions.size() < 0) { return false; }

	double max_error = std::numeric_limits<double>::infinity();

	Eigen::Matrix<double, 3, 3> Rd;
	Eigen::Matrix<double, 3, 1> td;

	Eigen::Matrix<double, 3, 1> dr;
	Eigen::Matrix<double, 3, 1> dt;

	for (auto const& solution : solutions)
	{
	Eigen::Matrix<double, 3, 3> Rv = Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v);
	Eigen::Matrix<double, 3, 1> tc = solution.C;

	double error = (u7.col(6) - ((Rv * X7.col(6)) + tc).colwise().hnormalized()).norm(); // TODO: use velocity
	if (error >= max_error) { continue; }
	max_error = error;

	Rd = Rv;
	td = tc;
	
	dr = solution.w;
	dt = solution.t;
	}

	Eigen::Matrix<double, 3, 1> r = vector_r_rodrigues(Rd * R_initial);
	Eigen::Matrix<double, 3, 1> t = td;

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	matrix_to_buffer(dr, dr_12);
	matrix_to_buffer(dt, dt_12);

	return is_valid_pose(r, t) && is_valid_pose(dr, dt);
}
