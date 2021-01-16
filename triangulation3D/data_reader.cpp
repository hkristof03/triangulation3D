#include "data_reader.h"


std::vector<std::string> ListDirectory(std::string& path)
{
    std::vector<std::string> file_paths;

    for (const auto& entry : fs::directory_iterator(path))
        file_paths.push_back(entry.path().string());

    return file_paths;
}


void ReadPoints(
    std::string path,
    std::vector<cv::Point2d>& points_img1,
    std::vector<cv::Point2d>& points_img2
)
{
    std::string line;
    std::ifstream file(path);

    std::cout << "Reading feature matched 2D point coordinates from path: "
        << std::endl << path << std::endl;

    // 3 x 48 = 144 points -> 3 chessboard
    size_t max_points_chessboard = 144;
    std::vector<std::vector<double>> coords;
    coords.reserve(4);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        double coord;
        std::vector<double> tmp;
        tmp.reserve(max_points_chessboard);
        
        while (ss >> coord)
        {
            tmp.emplace_back(coord);
        }
        coords.emplace_back(tmp);
    }
    file.close();

    const size_t n_points = coords.at(0).size();
    points_img1.clear(), points_img2.clear();
    points_img1.reserve(n_points), points_img2.reserve(n_points);

    for (size_t i = 0; i < n_points; ++i)
    {
        double
            &x1 = coords.at(0).at(i),
            &y1 = coords.at(1).at(i),
            &x2 = coords.at(2).at(i),
            &y2 = coords.at(3).at(i);

        points_img1.emplace_back(cv::Point2d(x1, y1));
        points_img2.emplace_back(cv::Point2d(x2, y2));
    }
    std::cout << "Finished reading from file. " << points_img1.size() 
        << " points are created." << std::endl;
}

cv::Mat ReadCameraMatrix(
    std::string path
)
{
    cv::Mat K;;
    std::ifstream file(path);
    double val;

    while (file >> val)
        K.push_back(val);

    K = K.reshape(1, 3);
    std::cout << "Camera matrix: " << std::endl << K << std::endl;

    return K;
}