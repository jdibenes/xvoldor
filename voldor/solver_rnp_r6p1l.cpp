
#include <limits>
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_r6p1l(float* p3d_1, float* p2d_2, bool direction, float r0, float* r_12, float* t_12)
{
	int const max_pow = 2;

	RSSinglelinCameraPoseVector solutions;

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X7 = matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 7);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u7 = matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, 7);
	
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X = X7(Eigen::indexing::all, Eigen::seqN(0, 6));
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u = u7(Eigen::indexing::all, Eigen::seqN(0, 6));

	R6P1Lin(X, u, direction, r0, max_pow, &solutions); // always returns 0

	if (solutions.size() < 0) { return false; }

	double max_error = std::numeric_limits<double>::infinity();

	Eigen::Matrix<double, 3, 1> r;
	Eigen::Matrix<double, 3, 1> t;

	for (auto const& solution : solutions)
	{
	Eigen::Matrix<double, 3, 3> Rv = matrix_R_rodrigues(solution.v);
	Eigen::Matrix<double, 3, 1> tc = solution.C;

	double error = (u7.col(6) - ((Rv * X7.col(6)) + tc).colwise().hnormalized()).norm();
	if (error >= max_error) { continue; }
	max_error = error;

	r = solution.v;
	t = tc;
	}

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	return is_valid_pose(r, t);
}
