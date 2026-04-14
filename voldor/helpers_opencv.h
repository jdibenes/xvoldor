
#pragma once

#include <opencv2/calib3d.hpp>

// Collect outputs
// p3d: [x, y, z]
// p2z: [u, v, z]
// p2k: [u, v]

// Solver inputs
// p3d: [x, y, z]
// p2h: [x, y, 1]
// p2d: [x, y]

inline cv::Point3f p3d_to_p2h(cv::Point3f const& p)
{
	cv::Point3f r{ p.x / p.z, p.y / p.z, 1 };
	return r;
}

inline cv::Point2f p3d_to_p2d(cv::Point3f const& p)
{
	cv::Point2f r{ p.x / p.z, p.y / p.z };
	return r;
}

inline cv::Point3f p2z_to_p3d(cv::Point3f const& p, float fx, float fy, float cx, float cy)
{
	cv::Point3f r{ (p.x - cx) * p.z / fx, (p.y - cy) * p.z / fy, p.z };
	return r;
}

inline cv::Point3f p2z_to_p2h(cv::Point3f const& p, float fx, float fy, float cx, float cy)
{
	cv::Point3f r{ (p.x - cx) / fx, (p.y - cy) / fy, 1 };
	return r;
}

inline cv::Point2f p2z_to_p2d(cv::Point3f const& p, float fx, float fy, float cx, float cy)
{
	cv::Point2f r{ (p.x - cx) / fx, (p.y - cy) / fy };
	return r;
}

inline cv::Point3f p2k_to_p2h(cv::Point2f const& p, float fx, float fy, float cx, float cy)
{
	cv::Point3f r{ (p.x - cx) / fx, (p.y - cy) / fy, 1 };
	return r;
}

inline cv::Point2f p2k_to_p2d(cv::Point2f const& p, float fx, float fy, float cx, float cy)
{
	cv::Point2f r{ (p.x - cx) / fx, (p.y - cy) / fy };
	return r;
}

inline bool is_valid_point(cv::Point3f const& p)
{
	return std::isfinite(p.x + p.y + p.z);
}

inline bool is_valid_point(cv::Point2f const& p)
{
	return std::isfinite(p.x + p.y);
}
