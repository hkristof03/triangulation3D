#include "planar_motion.h"


void normalizePoints(
	const std::vector<cv::Point2d>& input_source_points,
	const std::vector<cv::Point2d>& input_destination_points,
	std::vector<cv::Point2d>& output_source_points,
	std::vector<cv::Point2d>& output_destination_points,
	cv::Mat& T1,
	cv::Mat& T2)
{
	// The objective: normalize the point set in each image by
	// translating the mass point to the origin and
	// the average distance from the mass point to be sqrt(2).
	T1 = cv::Mat::eye(3, 3, CV_64F);
	T2 = cv::Mat::eye(3, 3, CV_64F);
	const size_t pointNumber = input_source_points.size();
	output_source_points.resize(pointNumber);
	output_destination_points.resize(pointNumber);
	// Calculate the mass point
	cv::Point2d mass1(0, 0), mass2(0, 0);

	for (auto i = 0; i < pointNumber; ++i)
	{
		mass1 = mass1 + input_source_points[i];
		mass2 = mass2 + input_destination_points[i];
	}

	mass1 = mass1 * (1.0 / pointNumber);
	mass2 = mass2 * (1.0 / pointNumber);
	// Translate the point clouds to the origin
	for (auto i = 0; i < pointNumber; ++i)
	{
		output_source_points[i] = input_source_points[i] - mass1;
		output_destination_points[i] = input_destination_points[i] - mass2;
	}
	// Calculate the average distances of the points from the origin
	double avgDistance1 = 0.0, avgDistance2 = 0.0;

	for (auto i = 0; i < pointNumber; ++i)
	{
		avgDistance1 += cv::norm(output_source_points[i]);
		avgDistance2 += cv::norm(output_destination_points[i]);
	}
	avgDistance1 /= pointNumber;
	avgDistance2 /= pointNumber;

	const double multiplier1 = sqrt(2) / avgDistance1;
	const double multiplier2 = sqrt(2) / avgDistance2;

	for (auto i = 0; i < pointNumber; ++i)
	{
		output_source_points[i] *= multiplier1;
		output_destination_points[i] *= multiplier2;
	}

	T1.at<double>(0, 0) = multiplier1;
	T1.at<double>(1, 1) = multiplier1;
	T1.at<double>(0, 2) = -multiplier1 * mass1.x;
	T1.at<double>(1, 2) = -multiplier1 * mass1.y;

	T2.at<double>(0, 0) = multiplier2;
	T2.at<double>(1, 1) = multiplier2;
	T2.at<double>(0, 2) = -multiplier2 * mass2.x;
	T2.at<double>(1, 2) = -multiplier2 * mass2.y;
	// Reason: T1 * point = (Scaling1 * Translation1) * point =
	// = Scaling1 * (Translation1 * point)
}

int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_)
{
	const double inlier_ratio =
		static_cast<double>(inlier_number_) / point_number_;

	static const double log1 = log(1.0 - confidence_);
	const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));

	const int k = log1 / log2;
	if (k < 0)
		return std::numeric_limits<int>::max();
	return k;
}

void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d>& source_points,
	const std::vector<cv::Point2d>& destination_points,
	cv::Mat& fundamental_matrix)
{
	const size_t pointNumber = source_points.size();
	cv::Mat A(pointNumber, 9, CV_64F);

	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		const double
			& x1 = source_points[pointIdx].x,
			& y1 = source_points[pointIdx].y,
			& x2 = destination_points[pointIdx].x,
			& y2 = destination_points[pointIdx].y;

		A.at<double>(pointIdx, 0) = x1 * x2;
		A.at<double>(pointIdx, 1) = x2 * y1;
		A.at<double>(pointIdx, 2) = x2;
		A.at<double>(pointIdx, 3) = y2 * x1;
		A.at<double>(pointIdx, 4) = y2 * y1;
		A.at<double>(pointIdx, 5) = y2;
		A.at<double>(pointIdx, 6) = x1;
		A.at<double>(pointIdx, 7) = y1;
		A.at<double>(pointIdx, 8) = 1;
	}
	cv::Mat evals, evecs;
	cv::Mat AtA = A.t() * A;
	cv::eigen(AtA, evals, evecs);

	cv::Mat x = evecs.row(evecs.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
	fundamental_matrix.create(3, 3, CV_64F);
	memcpy(fundamental_matrix.data, x.data, sizeof(double) * 9);
}

void ransacFundamentalMatrix(
	const std::vector<cv::Point2d>& input_src_points,
	const std::vector<cv::Point2d>& input_destination_points,
	const std::vector<cv::Point2d>& normalized_input_src_points,
	const std::vector<cv::Point2d>& normalized_input_destination_points,
	const cv::Mat& T1,
	const cv::Mat& T2,
	cv::Mat& fundamental_matrix,
	std::vector<size_t>& best_inliers,
	double confidence,
	double threshold)
{
	// The so-far-the-best fundamental matrix
	cv::Mat best_fundamental_matrix;
	// The number of correspondences
	const size_t point_number = input_src_points.size();
	// Initializing the index pool from which the minimal samples are selected
	std::vector<size_t> index_pool(point_number);
	for (size_t i = 0; i < point_number; ++i)
		index_pool[i] = i;

	// The size of a minimal sample
	constexpr size_t sample_size = 8;
	// The minimal sample
	size_t* mss = new size_t[sample_size];

	size_t maximum_iterations = std::numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
		iteration_limit = 5000, // A strict iteration limit which mustn't be exceeded
		iteration = 0; // The current iteration number

	std::vector<cv::Point2d> source_points(sample_size),
		destination_points(sample_size);

	while (iteration++ < MIN(iteration_limit, maximum_iterations))
	{
		for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
		{
			// Select a random index from the pool
			const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
			mss[sample_idx] = index_pool[idx];
			index_pool.erase(index_pool.begin() + idx);
			// Put the selected correspondences into the point containers
			const size_t point_idx = mss[sample_idx];
			source_points[sample_idx] = normalized_input_src_points[point_idx];
			destination_points[sample_idx] = normalized_input_destination_points[point_idx];
		}
		// Estimate fundamental matrix
		cv::Mat fundamental_matrix(3, 3, CV_64F);
		getFundamentalMatrixLSQ(source_points, destination_points, fundamental_matrix);
		fundamental_matrix = T2.t() * fundamental_matrix * T1; // Denormalize the fundamental matrix
		// Count the inliers
		std::vector<size_t> current_inliers;
		current_inliers.reserve(point_number);
		const double* p = (double*)fundamental_matrix.data;
		for (int i = 0; i < input_src_points.size(); ++i)
		{
			// Symmetric epipolar distance   
			cv::Mat pt1 = (cv::Mat_<double>(3, 1) 
				<< input_src_points[i].x, input_src_points[i].y, 1);
			cv::Mat pt2 = (cv::Mat_<double>(3, 1) 
				<< input_destination_points[i].x, input_destination_points[i].y, 1);
			// Calculate the error
			cv::Mat lL = fundamental_matrix.t() * pt2;
			cv::Mat lR = fundamental_matrix * pt1;
			// Calculate the distance of point pt1 from lL
			const double
				& aL = lL.at<double>(0),
				& bL = lL.at<double>(1),
				& cL = lL.at<double>(2);

			double tL = abs(aL * input_src_points[i].x + bL * input_src_points[i].y + cL);
			double dL = sqrt(aL * aL + bL * bL);
			double distanceL = tL / dL;
			// Calculate the distance of point pt2 from lR
			const double
				& aR = lR.at<double>(0),
				& bR = lR.at<double>(1),
				& cR = lR.at<double>(2);

			double tR = abs(aR * input_destination_points[i].x + bR * input_destination_points[i].y + cR);
			double dR = sqrt(aR * aR + bR * bR);
			double distanceR = tR / dR;

			double dist = 0.5 * (distanceL + distanceR);

			if (dist < threshold)
				current_inliers.push_back(i);
		}
		// Update if the new model is better than the previous so-far-the-best.
		if (best_inliers.size() < current_inliers.size())
		{
			// Update the set of inliers
			best_inliers.swap(current_inliers);
			current_inliers.clear();
			current_inliers.resize(0);
			// Update the model parameters
			best_fundamental_matrix = fundamental_matrix;
			// Update the iteration number
			maximum_iterations = getIterationNumber(point_number,
				best_inliers.size(),
				sample_size,
				confidence);
		}
		// Put back the selected points to the pool
		for (size_t i = 0; i < sample_size; ++i)
			index_pool.push_back(mss[i]);
	}
	delete[] mss;

	fundamental_matrix = best_fundamental_matrix;

	std::cout << "RANSAC finished! Number of inliers: " << best_inliers.size()
		<< std::endl << "Fundamental matrix: " << std::endl 
		<< best_fundamental_matrix << std::endl;
}

void linearTriangulation(
	const cv::Mat& projection_1,
	const cv::Mat& projection_2,
	const cv::Mat& src_point,
	const cv::Mat& dst_point,
	cv::Mat& point3d)
{
	cv::Mat A(4, 3, CV_64F);
	cv::Mat b(4, 1, CV_64F);

	{
		const double
			& px = src_point.at<double>(0),
			& py = src_point.at<double>(1),
			& p1 = projection_1.at<double>(0, 0),
			& p2 = projection_1.at<double>(0, 1),
			& p3 = projection_1.at<double>(0, 2),
			& p4 = projection_1.at<double>(0, 3),
			& p5 = projection_1.at<double>(1, 0),
			& p6 = projection_1.at<double>(1, 1),
			& p7 = projection_1.at<double>(1, 2),
			& p8 = projection_1.at<double>(1, 3),
			& p9 = projection_1.at<double>(2, 0),
			& p10 = projection_1.at<double>(2, 1),
			& p11 = projection_1.at<double>(2, 2),
			& p12 = projection_1.at<double>(2, 3);

		A.at<double>(0, 0) = px * p9 - p1;
		A.at<double>(0, 1) = px * p10 - p2;
		A.at<double>(0, 2) = px * p11 - p3;
		A.at<double>(1, 0) = py * p9 - p5;
		A.at<double>(1, 1) = py * p10 - p6;
		A.at<double>(1, 2) = py * p11 - p7;

		b.at<double>(0) = p4 - px * p12;
		b.at<double>(1) = p8 - py * p12;
	}
	{
		const double
			& px = dst_point.at<double>(0),
			& py = dst_point.at<double>(1),
			& p1 = projection_2.at<double>(0, 0),
			& p2 = projection_2.at<double>(0, 1),
			& p3 = projection_2.at<double>(0, 2),
			& p4 = projection_2.at<double>(0, 3),
			& p5 = projection_2.at<double>(1, 0),
			& p6 = projection_2.at<double>(1, 1),
			& p7 = projection_2.at<double>(1, 2),
			& p8 = projection_2.at<double>(1, 3),
			& p9 = projection_2.at<double>(2, 0),
			& p10 = projection_2.at<double>(2, 1),
			& p11 = projection_2.at<double>(2, 2),
			& p12 = projection_2.at<double>(2, 3);

		A.at<double>(2, 0) = px * p9 - p1;
		A.at<double>(2, 1) = px * p10 - p2;
		A.at<double>(2, 2) = px * p11 - p3;
		A.at<double>(3, 0) = py * p9 - p5;
		A.at<double>(3, 1) = py * p10 - p6;
		A.at<double>(3, 2) = py * p11 - p7;

		b.at<double>(2) = p4 - px * p12;
		b.at<double>(3) = p8 - py * p12;
	}

	//cv::Mat x = (A.t() * A).inv() * A.t() * b;
	point3d = A.inv(cv::DECOMP_SVD) * b;
}

void getProjectionMatrices(
	const cv::Mat& essential_matrix,
	const cv::Mat& K1,
	const cv::Mat& K2,
	const cv::Mat& src_point,
	const cv::Mat& dst_point,
	cv::Mat& projection_1,
	cv::Mat& projection_2)
{
	// ****************************************************
	// Calculate the projection matrix of the first camera
	// ****************************************************
	projection_1 = K1 * cv::Mat::eye(3, 4, CV_64F);

	// projection_1.create(3, 4, CV_64F);
	// cv::Mat rotation_1 = cv::Mat::eye(3, 3, CV_64F);
	// cv::Mat translation_1 = cv::Mat::zeros(3, 1, CV_64F);

	// ****************************************************
	// Calculate the projection matrix of the second camera
	// ****************************************************

	// Decompose the essential matrix
	cv::Mat rotation_1, rotation_2, translation;

	cv::SVD svd(essential_matrix, cv::SVD::FULL_UV);
	// It gives matrices U D Vt

	if (cv::determinant(svd.u) < 0)
		svd.u.col(2) *= -1;
	if (cv::determinant(svd.vt) < 0)
		svd.vt.row(2) *= -1;

	cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	rotation_1 = svd.u * w * svd.vt;
	rotation_2 = svd.u * w.t() * svd.vt;
	translation = svd.u.col(2) / cv::norm(svd.u.col(2));

	// The possible solutions:
	// (rotation_1, translation)
	// (rotation_2, translation)
	// (rotation_1, -translation)
	// (rotation_2, -translation)

	cv::Mat P21 = K2 * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P22 = K2 * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P23 = K2 * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), -translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), -translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), -translation.at<double>(2));
	cv::Mat P24 = K2 * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), -translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), -translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), -translation.at<double>(2));

	std::vector< const cv::Mat* > Ps = { &P21, &P22, &P23, &P24 };
	double minDistance = std::numeric_limits<double>::max();

	for (const auto& P2ptr : Ps)
	{
		const cv::Mat& P1 = projection_1;
		const cv::Mat& P2 = *P2ptr;

		// Estimate the 3D coordinates of a point correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			src_point,
			dst_point,
			point3d);
		point3d.push_back(1.0);

		cv::Mat projection1 = P1 * point3d;
		cv::Mat projection2 = P2 * point3d;

		if (projection1.at<double>(2) < 0 ||
			projection2.at<double>(2) < 0)
			continue;

		projection1 = projection1 / projection1.at<double>(2);
		projection2 = projection2 / projection2.at<double>(2);

		// cv::norm(projection1 - src_point)
		double dx1 = projection1.at<double>(0) - src_point.at<double>(0);
		double dy1 = projection1.at<double>(1) - src_point.at<double>(1);
		double squaredDist1 = dx1 * dx1 + dy1 * dy1;

		// cv::norm(projection2 - dst_point)
		double dx2 = projection2.at<double>(0) - dst_point.at<double>(0);
		double dy2 = projection2.at<double>(1) - dst_point.at<double>(1);
		double squaredDist2 = dx2 * dx2 + dy2 * dy2;

		if (squaredDist1 + squaredDist2 < minDistance)
		{
			minDistance = squaredDist1 + squaredDist2;
			projection_2 = P2.clone();
		}
	}
}

void printTimes(const std::chrono::time_point<std::chrono::system_clock>& start,
	const std::chrono::time_point<std::chrono::system_clock>& end,
	const std::string& message)
{
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("Processing time of the %s was %f secs.\n", message.c_str(), elapsed_seconds.count());
}