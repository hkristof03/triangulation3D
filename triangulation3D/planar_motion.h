#pragma once
#include <iostream>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>  

// Normalizing the point coordinates for the fundamental matrix estimation
void normalizePoints(
	const std::vector<cv::Point2d>& input_source_points, // Points in the source image
	const std::vector<cv::Point2d>& input_destination_points, // Points in the destination image
	std::vector<cv::Point2d>& output_source_points, // Normalized points in the source image
	std::vector<cv::Point2d>& output_destination_points, // Normalized points in the destination image
	cv::Mat& T1, // Normalizing transformation in the source image
	cv::Mat& T2); // Normalizing transformation in the destination image

// Return the iteration number of RANSAC given the inlier ratio and
// a user-defined confidence value.
int getIterationNumber(
	int point_number, // The number of points
	int inlier_number, // The number of inliers
	int sample_size, // The sample size
	double confidence); // The required confidence

// A function estimating the fundamental matrix from point correspondences
// by least-squares fitting.
void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d>& source_points_, // Points in the source image
	const std::vector<cv::Point2d>& destination_points_, // Points in the destination image
	cv::Mat& fundamental_matrix); // The estimated fundamental matrix

// A function estimating the fundamental matrix from point correspondences
// by RANSAC.
void ransacFundamentalMatrix(
	const std::vector<cv::Point2d>& input_source_points, // Points in the source image
	const std::vector<cv::Point2d>& input_destination_points, // Points in the destination image
	const std::vector<cv::Point2d>& normalized_input_src_points, // Normalized points in the source image
	const std::vector<cv::Point2d>& normalized_input_destination_points, // Normalized points in the destination image
	const cv::Mat& T1, // Normalizing transformation in the source image
	const cv::Mat& T2, // Normalizing transformation in the destination image
	cv::Mat& fundamental_matrix, // The estimated fundamental matrix
	std::vector<size_t>& inliers, // The inliers of the fundamental matrix
	double confidence, // The required confidence of RANSAC
	double threshold); // The inlier-outlier threshold

// A function estimating the 3D point coordinates from a point correspondences
// from the projection matrices of the two observing cameras.
void linearTriangulation(
	const cv::Mat& projection_1, // The projection matrix of the source image
	const cv::Mat& projection_2, // The projection matrix of the destination image
	const cv::Mat& src_point, // A point in the source image
	const cv::Mat& dst_point, // A point in the destination image
	cv::Mat& point3d); // The estimated 3D coordinates

// A function decomposing the essential matrix to the projection matrices
// of the two cameras.
void getProjectionMatrices(
	const cv::Mat& essential_matrix, // The parameters of the essential matrix
	const cv::Mat& K1, // The intrinsic camera parameters of the source image
	const cv::Mat& K2, // The intrinsic camera parameters of the destination image
	const cv::Mat& src_point, // A point in the source image
	const cv::Mat& dst_point, // A point in the destination image
	cv::Mat& projection_1, // The projection matrix of the source image
	cv::Mat& projection_2); // The projection matrix of the destination image

// Printing the time to the console
void printTimes(
	const std::chrono::time_point<std::chrono::system_clock>& start, // The starting time
	const std::chrono::time_point<std::chrono::system_clock>& end, // The current time
	const std::string& message); // The message to be written

// Visualize the effect of the point normalization
void checkEffectOfNormalization(const std::vector<cv::Point2d>& source_points_,  // Points in the first image 
	const std::vector<cv::Point2d>& destination_points_,   // Points in the second image
	const std::vector<cv::Point2d>& normalized_source_points_,  // Normalized points in the first image 
	const std::vector<cv::Point2d>& normalized_destination_points_, // Normalized points in the second image
	const cv::Mat& T1, // Normalizing transforcv::Mation in the first image
	const cv::Mat& T2, // Normalizing transforcv::Mation in the second image
	const std::vector<size_t>& inliers); // The inliers of the fundamental matrix