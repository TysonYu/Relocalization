//
// Created by Tiezheng Yu on 3/11/2019
//

#ifndef SOLVEPNP_H
#define SOLVEPNP_H

#include <opencv2/opencv.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>//allows us to use pcl::transformPointCloud function
#include <pcl/filters/passthrough.h>//allows us to use pcl::PassThrough

#include <LoadData.h>
#include <iostream>
#include <algorithm>

class Solver
{
public:
    Calibration calibration;

    std::string camera_time_stamp;
    //-- ground truth 是temp2World ----------------------------------------------
    cv::Mat rt_body2temp = (cv::Mat_<float>(4, 4) 
        << 1.0001574 , 0.062081739 , 0.0030256696 , -0.015831091 , 
        -0.062114749 , 1.0020664 , -0.074428819 , -0.028843965 , 
        -0.013841577 , 0.07346423 , 1.0018846 , -7.7168137e-05 , 
        0 , 0 , 0 , 1);
    cv::Mat rt_cam02body = (cv::Mat_<float>(4, 4)
        << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0);
    pcl::PointCloud<pcl::PointXYZRGB> world_cloud;
    std::vector<cv::Point2f> test_key_points;
    std::vector<cv::Point2f> dataset_key_points;
    std::vector<cv::Point3f> key_points_location;
    pcl::PointCloud<pcl::PointXYZRGB> result_cloud;
    float max_depth = 0;
    cv::Mat R;
    cv::Mat T;
    Solver(const std::string &camera_time_stamp, std::vector<cv::Point2f> test_key_points, std::vector<cv::Point2f> dataset_key_points):camera_time_stamp(camera_time_stamp),test_key_points(test_key_points),dataset_key_points(dataset_key_points)
    {
        LoadPointCloud();
        Get3DPoints();
        ShowResults();
    }

    void LoadPointCloud();
    void Get3DPoints();
    void ShowResults();
};



#endif //SOLVEPNP_H