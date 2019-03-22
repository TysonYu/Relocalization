//
// create by Tiezheng YU on 3/10/2019
//

#ifndef LOADDATA_H
#define LOADDATA_H

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

class Calibration
{
public:
    float camera_width;
    float camera_height;
//---------cam0---------------------------------------------
    float camera0_fu;
    float camera0_fv;
    float camera0_cu;
    float camera0_cv;
    float camera0_k1;
    float camera0_k2;
    float camera0_p1;
    float camera0_p2;
    float distortion_coefficients0[4];
    cv::Mat rt_cam02body;
//---------cam1---------------------------------------------
    float camera1_fu;
    float camera1_fv;
    float camera1_cu;
    float camera1_cv;
    float camera1_k1;
    float camera1_k2;
    float camera1_p1;
    float camera1_p2;
    float distortion_coefficients1[4];
    cv::Mat rt_cam12body;

    Calibration();
    void GetCalibration(const std::string &camera_calibration0,const std::string &camera_calibration1);
};

class Name
{
public:

    std::vector<std::string> timestamps; //声明一个字符串向量

    Name();
    void GetTimetamp();
};

class GroundTruth
{
public:
    std::vector<std::string> groundtruth;
    std::string camera_time_stamp;

    GroundTruth(const std::string &camera_time_stamp):camera_time_stamp(camera_time_stamp)
    {
        LoadGroundTruth();
    }
    void LoadGroundTruth();
};

class TestName
{
public:
    std::vector<std::string> timestamps;
    TestName()
    {
        LoadTimestamps();
    }
    void LoadTimestamps();
    std::string GetImageName(int test_id);
};

// class TestGroundTruth
// {
// public:
//     std::vector<std::string> groundtruth;
//     std::string camera_time_stamp;
//     Ground()
// }
#endif //LOADDATA_H