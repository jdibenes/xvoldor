
#include <opencv2/calib3d.hpp>
#include <rnp/rnp.h>
#include "solvers.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

// OK
bool solver_rpf_r7pfi(float const* p3d_1, float const* p2k_2, float cx, float cy, bool direction, float r0, float* r_12, float* t_12, float* dr_12, float* dt_12, float* f_xy, int max_iterations)
{
	float r_initial[3];
	float t_initial[3];
	float f_initial[2];

	bool ok;

	ok = solver_ppf_p4pf(p3d_1, p2k_2, true, cx, cy, r_initial, t_initial, f_initial);
	if (!ok) { return false; }

	Eigen::Matrix<double, 3, 3> R_initial = matrix_R_rodrigues(matrix_from_buffer<double, 3, 1>(r_initial));
	Eigen::Matrix<double, 2, 1> pp{ static_cast<double>(cx), static_cast<double>(cy) };

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X7 = R_initial * matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, 7);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u7 =             matrix_from_buffer<double, Eigen::Dynamic, Eigen::Dynamic>(p2k_2, 2, 7).colwise() - pp;

	RSDoublelinCameraPose solution;

	ok = !iterativeRnP<RSDoublelinCameraPose, R7PfIter>(X7, u7, Eigen::Vector3d{ 0, 0, 0 }, 7, r0, direction, max_iterations, solution); // vk, sampleSize not used
	if (!ok) { return false; }

	Eigen::Matrix<double, 3, 1> r = vector_r_rodrigues((Eigen::Matrix<double, 3, 3>::Identity() + matrix_cross(solution.v)) * R_initial);
	Eigen::Matrix<double, 3, 1> t = solution.C;

	Eigen::Matrix<double, 3, 1> dr = solution.w;
	Eigen::Matrix<double, 3, 1> dt = solution.t;

	Eigen::Matrix<double, 2, 1> f{ solution.f, solution.f };

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);

	matrix_to_buffer(dr, dr_12);
	matrix_to_buffer(dt, dt_12);

	matrix_to_buffer(f, f_xy);

	return is_valid_pose(r, t) && is_valid_pose(dr, dt) && is_valid_focal(f);
}
