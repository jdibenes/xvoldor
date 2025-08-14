
#include <opencv2/opencv.hpp>
#include "kitti.h"
#include "../gpu-kernels/gpu_kernels.h"

KittiGround
estimate_kitti_ground_plane
(
	cv::Mat depth,
	cv::Rect roi,
	cv::Mat K,
	int holo_width,
	float ms_kernel_var
)
{
	int const w = depth.cols;
	int const h = depth.rows;

	cv::Mat K_inv = K.inv();

	float* ground_params = new float[roi.width * roi.height * 4]; // h, n1,n2,n3
	float* height_params = new float[roi.width * roi.height]; // h (for evaluate norm scale for height)
	int valid_N = 0;

	cv::Point3f* p3_buffer = new cv::Point3f[(holo_width * 2 + 1) * (holo_width * 2 + 1)];

	for (int y = roi.y; y < roi.y + roi.height; y++) {
		for (int x = roi.x; x < roi.x + roi.width; x++) {

			cv::Point3f mean(0, 0, 0);

			int N = 0;
			for (int ky = -holo_width; ky <= holo_width; ky++) {
				for (int kx = -holo_width; kx <= holo_width; kx++) {
					// fall in image
					if (x + kx >= 0 && x + kx < w && y + ky >= 0 && y + ky < h) {
						cv::Point3f p3(x + kx, y + ky, 1);
						p3 = cv::Point3f(p3.dot(K_inv.at<cv::Point3f>(0)),
							p3.dot(K_inv.at<cv::Point3f>(1)), p3.dot(K_inv.at<cv::Point3f>(2))) * depth.at<float>(y + ky, x + kx);
						p3_buffer[N++] = p3;
						mean += p3;
					}
				}
			}

			mean /= N;
			float cov[6]{ 0,0,0,0,0,0 };
			for (int i = 0; i < N; i++) {
				cv::Point3f p3 = p3_buffer[i];
				cov[0] += (p3.x - mean.x) * (p3.x - mean.x);
				cov[1] += (p3.x - mean.x) * (p3.y - mean.y);
				cov[2] += (p3.x - mean.x) * (p3.z - mean.z);
				cov[3] += (p3.y - mean.y) * (p3.y - mean.y);
				cov[4] += (p3.y - mean.y) * (p3.z - mean.z);
				cov[5] += (p3.z - mean.z) * (p3.z - mean.z);
			}
			cv::Matx33f cov_mat(cov[0], cov[1], cov[2],
				cov[1], cov[3], cov[4],
				cov[2], cov[4], cov[5]);

			cv::Matx31f eval;
			cv::Matx33f evec;
			if (eigen(cov_mat, eval, evec)) {

				cv::Vec3f n = cv::Vec3f(evec.val[6], evec.val[7], evec.val[8]);
				n /= norm(n, cv::NORM_L2);
				cv::Point3f p3(x, y, 1);
				p3 = cv::Point3f(p3.dot(K_inv.at<cv::Point3f>(0)),
					p3.dot(K_inv.at<cv::Point3f>(1)), p3.dot(K_inv.at<cv::Point3f>(2))) * depth.at<float>(y, x);

				float height = n.dot(p3);
				if (!isfinite(height))
					continue;

				if (height > 0)
					n = -n; // make normal vector point to view point
				else
					height = -height; // make height positive
				ground_params[valid_N * 4 + 0] = height;
				ground_params[valid_N * 4 + 1] = n.val[0];
				ground_params[valid_N * 4 + 2] = n.val[1];
				ground_params[valid_N * 4 + 3] = n.val[2];
				height_params[valid_N] = height;
				valid_N++;
			}
		}
	}

	KittiGround ret;

	if (valid_N < 1)
		return ret;

	// normalize with height median
	std::sort(height_params, height_params + valid_N);
	ret._height_median = height_params[valid_N / 2];
	for (int i = 0; i < valid_N; i++)
		ground_params[i * 4] /= ret._height_median;



	cv::Vec4f mean(1, 0, -1, 0);
	meanshift_gpu(ground_params, ms_kernel_var, mean.val, &ret.confidence, &ret.used_iters, true, valid_N, 4);
	ret.height = mean[0] * ret._height_median;
	ret.normal = cv::Vec3f(mean[1], mean[2], mean[3]);

	delete[] ground_params;
	delete[] height_params;
	delete[] p3_buffer;
	return ret;
}
