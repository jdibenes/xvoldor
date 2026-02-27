
#include <opencv2/calib3d.hpp>
#include "solver_gpm.h"
#include "batch_solve_common.h"

// points in format [u, v, z]
int batch_solve_gpm_hpc2_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations)
{
	int n_points        = (int)pts0.size();
	int poses_pool_used = 0;

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	cv::Vec3f r;
	cv::Vec3f t;

	for (int i = 0; i < poses_to_sample; ++i)
	{
		int ix[3];

		sample(n_points, 3, ix);

		int i1 = ix[0];
		int i2 = ix[1];
		int i3 = ix[2];

		cv::Point3f p1[3];
		cv::Point3f p2[3];

		p1[0] = pts0[i1];
		p1[1] = pts0[i2];
		p1[2] = pts0[i3];
		p2[0] = pts1[i1];
		p2[1] = pts1[i2];
		p2[2] = pts1[i3];

		cv::Point3f& pa1 = p1[0];
		cv::Point3f& pb1 = p1[1];
		cv::Point3f& pc1 = p1[2];
		cv::Point3f& pa2 = p2[0];
		cv::Point3f& pb2 = p2[1];
		cv::Point3f& pc2 = p2[2];

		pa1.x = ((pa1.x - cx) / fx) * pa1.z;
		pb1.x = ((pb1.x - cx) / fx) * pb1.z;
		pc1.x = ((pc1.x - cx) / fx) * pc1.z;
		pa2.x = ((pa2.x - cx) / fx) * pa2.z;
		pb2.x = ((pb2.x - cx) / fx) * pb2.z;
		pc2.x = ((pc2.x - cx) / fx) * pc2.z;

		pa1.y = ((pa1.y - cy) / fy) * pa1.z;
		pb1.y = ((pb1.y - cy) / fy) * pb1.z;
		pc1.y = ((pc1.y - cy) / fy) * pc1.z;
		pa2.y = ((pa2.y - cy) / fy) * pa2.z;
		pb2.y = ((pb2.y - cy) / fy) * pb2.z;
		pc2.y = ((pc2.y - cy) / fy) * pc2.z;

		bool ok = solver_gpm_hpc2((float*)p1, (float*)p2, (float*)&r, (float*)&t, refine_iterations);
		if (!ok) { continue; }

		poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = r;
		poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = t;

		poses_pool_used++;
	}

	return poses_pool_used;
}
