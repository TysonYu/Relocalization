//
// create by Tiezheng YU on 3/15/2019
//

#ifndef KEYPOINTSMAKER_H
#define KEYPOINTSMAKER_H

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


class KeyPointsMaker
{
public:
    //--------- 去畸变 ------------------------------------------
    Calibration calibration;
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1);//内参矩阵
    cv::Mat distortion_coefficients = cv::Mat(4,1, CV_32FC1);//畸变矩阵
    //--------- 训练集图像 ---------------------------------------
    Name TrainSetName;//训练集图像的名字
    std::vector<cv::Mat> images;//储存所有训练集图像（未去畸变）
    //--------- 储存所有训练集图像上对应特征点的vector<Point2f> -------
    std::vector<cv::KeyPoint> train_keypoints;
    std::vector<cv::Point2f> train_key_points_2d;
    //--------- 得到所有特征点的3D坐标-------------------------------
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
    std::vector<cv::Point3f> train_key_points_3d;
    pcl::PointCloud<pcl::PointXYZRGB> result_cloud;
    float max_depth = 0;
    //--------- 储存每个keypoint的descriptor ----------------------
    cv::Mat train_descriptor;
    


    KeyPointsMaker()
    {
        GetCameraMatrix();
        LoadImage();
        LoadPointCloud();
        // for(int i = 1140; i < TrainSetName.timestamps.size()/20; i++)
        std::cout << "Adding train set KeyPoints ......" << std::endl;
        for(int i = 50; i < 2800; i = i+10)
        {
            AddKeyPoints(i);
        }

    }
    void GetCameraMatrix();
    void LoadImage();
    void LoadPointCloud();
    void AddKeyPoints(int train_image_number);
    void ShowKeyPointCloud();
};

#endif //KEYPOINTSMAKER_H