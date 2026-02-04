
#include "../rolling_shutter/rnp.h"
#include <iostream>
#include "helpers_eigen.h"
#include "helpers_geometry.h"



bool solver_r6p1l(float const* p3d, float const* p2d, bool direction, float r0, int maxpow, float *r01, float* t01)
{
	RSSinglelinCameraPoseVector solutions;

	Eigen::MatrixXd X7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d, 3, 7).cast<double>();
	Eigen::MatrixXd u7 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d, 2, 7).cast<double>();
	
	Eigen::MatrixXd X = X7(Eigen::all, Eigen::seqN(0, 6));
	Eigen::MatrixXd u = u7(Eigen::all, Eigen::seqN(0, 6));

	R6P1Lin(X, u, direction, r0, maxpow, &solutions);

	if (solutions.size() < 0) { return false; }

	double max_error = HUGE_VAL;
	Eigen::Matrix<float, 3, 3> R;
	Eigen::Matrix<float, 3, 1> t;


	for (auto const& solution : solutions) 
	{
		//Eigen::Matrix<double, 3, 3> Rv = matrix_R_cayley<double, 3, 3>(solution.v(0), solution.v(1), solution.v(2));
		Eigen::Matrix<double, 3, 3> Rv = Eigen::AngleAxis<double>(solution.v.norm(), solution.v.normalized()).toRotationMatrix();
		double error = (u7.col(6) - (Rv * X7.col(6) + solution.C).colwise().hnormalized()).norm();
		std::cout << "RV" << std::endl;
		std::cout << Rv << std::endl;
		std::cout << solution.v << std::endl;
		std::cout << solution.C << std::endl;
		std::cout << solution.t << std::endl;
		std::cout << solution.w << std::endl;
		if (error >= max_error) { continue; }
		R = Rv.cast<float>();
		t = solution.C.cast<float>();
		max_error = error;
	}

	Eigen::AngleAxis<float> aa(R);

	Eigen::Matrix<float, 3, 1> r = aa.axis() * aa.angle();

	matrix_to_buffer(r, r01);
	matrix_to_buffer(t, t01);

	return true;
}

bool solver_r6p2l()
{

}

bool solver_r6pi()
{

}

bool solver_r7pf()
{

}
