
#include <limits>
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p1l(float const* p3d_1, float const* p2d_2, bool direction, float r0, int max_pow, float* r_12, float* t_12)
{
	RSSinglelinCameraPoseVector solutions;

	Eigen::MatrixXd X7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 7).cast<double>();
	Eigen::MatrixXd u7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, 7).cast<double>();
	
	Eigen::MatrixXd X = X7(Eigen::indexing::all, Eigen::seqN(0, 6));
	Eigen::MatrixXd u = u7(Eigen::indexing::all, Eigen::seqN(0, 6));

	R6P1Lin(X, u, direction, r0, max_pow, &solutions); // always returns 0

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

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
