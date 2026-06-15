
#include <PoseLib/robust.h>
#include "helpers_camera.h"

void estimate_depth_closed_form(cv::Mat const& flow, cv::Mat const& K, cv::Mat const& K_inv, cv::Mat const& R, cv::Mat const& t, cv::Mat& depth, float min_depth, float max_depth)
{
	cv::Mat b = K * t;
	cv::Mat KRKinv = K * R * K_inv;

	float b1 = b.at<float>(0);
	float b2 = b.at<float>(1);
	float b3 = b.at<float>(2);

	for (int y = 0; y < flow.rows; y++)
	{
	for (int x = 0; x < flow.cols; x++)
	{
	cv::Point2f delta = flow.at<cv::Point2f>(y, x);
	cv::Mat P = (cv::Mat_<float>(3, 1) << x, y, 1);

	P = KRKinv * P;

	float w1 = P.at<float>(0);
	float w2 = P.at<float>(1);
	float w3 = P.at<float>(2);

	float a1 = x + delta.x;
	float a2 = y + delta.y;

	float z_nume = (a1 * b3 - b1) * (w1 - a1 * w3) + (a2 * b3 - b2) * (w2 - a2 * w3);
	float z_deno = (w1 - a1 * w3) * (w1 - a1 * w3) + (w2 - a2 * w3) * (w2 - a2 * w3);

	depth.at<float>(y, x) = fminf(fmaxf(z_nume / z_deno, min_depth), max_depth);
	}
	}
}

void estimate_camera_pose_epipolar(cv::Mat const& flow, cv::Mat const& K, cv::Mat& E, cv::Mat& R, cv::Mat& t, cv::Mat const& mask, int sampling_2d_step)
{
	int const w = flow.cols;
	int const h = flow.rows;

	bool const use_external_mask = !mask.empty();

	cv::Mat pts1(w * h, 2, CV_32F);
	cv::Mat pts2(w * h, 2, CV_32F);
	
	float* pts1_iter = reinterpret_cast<float*>(pts1.data);
	float* pts2_iter = reinterpret_cast<float*>(pts2.data);

	int N_used = 0;

	for (int y = 0; y < h; y += sampling_2d_step)
	{
	for (int x = 0; x < w; x += sampling_2d_step)
	{
	if (use_external_mask && mask.at<float>(y, x) < 0.5) { continue; }
			
	*pts1_iter++ = x + 0.0f;
	*pts1_iter++ = y + 0.0f;
	*pts2_iter++ = x + flow.at<cv::Vec2f>(y, x)[0];
	*pts2_iter++ = y + flow.at<cv::Vec2f>(y, x)[1];

	N_used++;
	}
	}

	pts1 = pts1.rowRange(0, N_used);
	pts2 = pts2.rowRange(0, N_used);

	cv::Mat pts_mask;

	E = cv::findEssentialMat(pts1, pts2, K, cv::LMEDS, 0.999, 1.0, pts_mask);

	cv::recoverPose(E, pts1, pts2, K, R, t);

	E.convertTo(E, CV_32F);
	R.convertTo(R, CV_32F);
	t.convertTo(t, CV_32F);
}

void estimate_camera_focal(cv::Mat const& flow, float& fx, float& fy, float cx, float cy, int sampling_2d_step)
{
	int const w = flow.cols;
	int const h = flow.rows;

	std::vector<poselib::Point2D> p2k_1;
	std::vector<poselib::Point2D> p2k_2;

	for (int y = 0; y < h; y += sampling_2d_step)
	{
	for (int x = 0; x < w; x += sampling_2d_step)
	{
	p2k_1.push_back({ x, y });
	p2k_2.push_back({ x + flow.at<cv::Vec2f>(y, x)[0], y + flow.at<cv::Vec2f>(y, x)[1] });
	}
	}

	poselib::Point2D pp{ cx, cy };

	poselib::RelativePoseOptions rpo;
	poselib::ImagePair ip;
	std::vector<char> inliers;

	poselib::RansacStats rs = poselib::estimate_shared_focal_relative_pose(p2k_1, p2k_2, pp, rpo, &ip, &inliers);

	fx = static_cast<float>(ip.camera1.focal_x());
	fy = static_cast<float>(ip.camera1.focal_y());
}
