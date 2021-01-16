#pragma once
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "data_reader.h"
#include "planar_motion.h"


int main(int argc, const char** argv)
{
    srand(time(NULL));
    // Using time point and system_clock 
    std::chrono::time_point<std::chrono::system_clock> start, end;

    std::string path_data = argv[1];
    std::string path_result = argv[2];

    std::vector<std::string> file_paths = ListDirectory(path_data);

    for (auto& fp : file_paths)
    {
        cv::Mat
            img1,
            img2,
            T1,
            T2,
            F, // Fundamental matrix
            K, // calibrated camera parameters
            E, // Essential matrix
            P1, // Projection matrix corresponding to the 1st image
            P2; // Projection matrix corresponding to the 2nd image
        std::vector<cv::Point2d>
            points_img1,  // Point correspondences first image
            points_img2, // Point correspondences second image
            normalized_points_img1, // Point correspondences first image
            normalized_points_img2; // Point correspondences second image
        std::vector<size_t> best_inliers;
        double
            confidence = 0.99,
            threshold = 1.0;

        std::vector<std::string> files = ListDirectory(fp);
        
        start = std::chrono::system_clock::now();
        for (const auto& it : files)
        {
            if (it.find(".png") != std::string::npos)
            {
                if (img1.empty())
                    img1 = cv::imread(it);
                else
                    img2 = cv::imread(it);
            }
            else if (it.find("camera.txt") != std::string::npos)
                K = ReadCameraMatrix(it);
            else
                ReadPoints(it, points_img1, points_img2);
        }
        end = std::chrono::system_clock::now();
        printTimes(start, end, "reading of images, points and camera parameters");

        start = std::chrono::system_clock::now();
        normalizePoints(points_img1, points_img2, normalized_points_img1,
            normalized_points_img2, T1, T2);
        end = std::chrono::system_clock::now();
        printTimes(start, end, "feature normalization");

        start = std::chrono::system_clock::now();
        ransacFundamentalMatrix(points_img1, points_img2, normalized_points_img1,
            normalized_points_img2, T1, T2, F, best_inliers, confidence, threshold);
        end = std::chrono::system_clock::now();
        printTimes(start, end, "estimation of the Fundamental matrix with RANSAC");

        // Essential matrix
        E = K.t() * F * K;

        getProjectionMatrices(E, K, K, (cv::Mat)points_img1[best_inliers[0]],
            (cv::Mat)points_img2[best_inliers[0]], P1, P2);

        // Draw the points and the corresponding epipolar lines
        constexpr double resize_ratio = 2.0;
        cv::Mat tmp_image1, tmp_image2;
        resize(img1, tmp_image1, cv::Size(img1.cols / resize_ratio, img1.rows / resize_ratio));
        resize(img2, tmp_image2, cv::Size(img2.cols / resize_ratio, img2.rows / resize_ratio));

        std::vector<cv::KeyPoint> src_inliers, dst_inliers;
        std::vector<cv::DMatch> inlier_matches;
        src_inliers.reserve(best_inliers.size());
        dst_inliers.reserve(best_inliers.size());
        inlier_matches.reserve(best_inliers.size());
        std::ofstream out_file(fp + "\\triangulated_points.txt");

        start = std::chrono::system_clock::now();

        for (auto inl_idx = 0; inl_idx < best_inliers.size(); inl_idx += 1)
        {
            const size_t& inlierIdx = best_inliers[inl_idx];
            const cv::Mat pt1 = static_cast<cv::Mat>(points_img1[inlierIdx]);
            const cv::Mat pt2 = static_cast<cv::Mat>(points_img2[inlierIdx]);

            // Estimate the 3D coordinates of the current inlier correspondence
            cv::Mat point3d;
            linearTriangulation(P1, P2, pt1, pt2, point3d);

            // Get the color of the point
            const int xi1 = round(points_img1[inlierIdx].x);
            const int yi1 = round(points_img1[inlierIdx].y);
            const int xi2 = round(points_img2[inlierIdx].x);
            const int yi2 = round(points_img2[inlierIdx].y);

            const cv::Vec3b& color1 = (cv::Vec3b)img1.at<cv::Vec3b>(yi1, xi1);
            const cv::Vec3b& color2 = (cv::Vec3b)img2.at<cv::Vec3b>(yi2, xi2);
            const cv::Vec3b color = 0.5 * (color1 + color2);

            const int blue = color[0];
            const int green = color[1];
            const int red = color[2];

            out_file << point3d.at<double>(0) << " "
                << point3d.at<double>(1) << " "
                << point3d.at<double>(2) << " "
                << blue << " "
                << green << " "
                << red << "\n";

            src_inliers.emplace_back(cv::KeyPoint());
            dst_inliers.emplace_back(cv::KeyPoint());
            inlier_matches.emplace_back(cv::DMatch());

            // Construct the cv::Matches std::vector for the drawing
            src_inliers.back().pt = points_img1[best_inliers[inl_idx]] / resize_ratio;
            dst_inliers.back().pt = points_img2[best_inliers[inl_idx]] / resize_ratio;
            inlier_matches.back().queryIdx = src_inliers.size() - 1;
            inlier_matches.back().trainIdx = dst_inliers.size() - 1;
        }
        out_file.close();

        cv::Mat out_image;
        drawMatches(tmp_image1, src_inliers, tmp_image2, dst_inliers, inlier_matches, out_image);

        cv::imshow("Matches", out_image);
        cv::waitKey(0);
    }

    return 0;
}


