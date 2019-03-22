//
// Created by Tiezheng Yu on 3/15/2019
//

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/opencv.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>//allows us to use pcl::transformPointCloud function
#include <pcl/filters/passthrough.h>//allows us to use pcl::PassThrough

#include <LoadData.h>
#include <KeyPointsMaker.h>

#include <iostream>
#include <algorithm>

class PNPSolver
{
public:
    KeyPointsMaker KeyPoints;
    TestName TestImageName;
    cv::Mat R;
    cv::Mat T;
    pcl::PointCloud<pcl::PointXYZRGB> result_cloud;
    PNPSolver(){};
    void FindMatches(int test_image_number);
    void ShowResults();
};



#endif //PNPSOLVER_H