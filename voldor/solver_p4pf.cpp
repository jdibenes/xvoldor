
#include <PoseLib/solvers/p4pf.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_p4pf(float* p3d_1, float* p2d_2, float cx, float cy, float* r_12, float* t_12, float* focal)
{
	std::vector<Eigen::Matrix<double, 3, 1>> P1;
	std::vector<Eigen::Matrix<double, 2, 1>> P2;

	Eigen::Matrix<double, 2, 1> pp = { cx, cy };

	for (int i = 0; i < 5; ++i)
	{
	P1.push_back(matrix_from_buffer<double, 3, 1>(p3d_1 + (i * 3)));
	P2.push_back(matrix_from_buffer<double, 2, 1>(p2d_2 + (i * 2)) - pp);
	}

	poselib::CameraPoseVector solutions;
	std::vector<double> focals;

	int count = poselib::p4pf(P2, P1, &solutions, &focals);
	if (count <= 0) { return false; }

	double max_error = std::numeric_limits<double>::infinity();

	Eigen::Matrix<double, 3, 3> Rd;
	Eigen::Matrix<double, 3, 1> td;
	double fd;

	for (int i = 0; i < count; ++i)
	{
	Eigen::Matrix<double, 3, 3> Rv = solutions[i].R();
	Eigen::Matrix<double, 3, 1> tv = solutions[i].t;
	double fv = focals[i];

	double error = (P2[5] - (fv * ((Rv * P1[5]) + tv).colwise().hnormalized())).norm();
	if (error >= max_error) { continue; }
	max_error = error;

	Rd = Rv;
	td = tv;
	fd = fv;
	}

	Eigen::Matrix<double, 3, 1> r = vector_r_rodrigues(Rd);
	Eigen::Matrix<double, 3, 1> t = td;
	float f = fd;

	matrix_to_buffer(r, r_12);
	matrix_to_buffer(t, t_12);
	focal[0] = f;

	return is_valid_pose(r, t) && is_valid_focal(f);
}
