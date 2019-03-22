//
// Greated by Tiezheng YU on 3/12/2019
//

#ifndef LOOPDETECTOR_H
#define LOOPDETECTOR_H

#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <LoadData.h>

class LoopDetector
{
public:
    Name names;
    DBoW3::Vocabulary vocab;
    std::vector<cv::Mat> images;//训练集所有图像
    std::vector<cv::Mat> descriptors;//练集的所有descriptors
    cv::Ptr< cv::Feature2D > detector = cv::ORB::create();
    int id_finded;
    std::string input_image_name;
    LoopDetector()
    {
        LoadVocabulary();
        LoadImage();
        CreateFeatures();
    };
    void LoadVocabulary();
    void LoadImage();
    void CreateFeatures();
    int FindID(std::string test_name);
};


#endif //LOOPDETECTOR_H