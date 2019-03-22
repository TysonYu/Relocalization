//
// create by Tiezheng YU on 3/10/2019
//

#include <ImageMatcher.h>

using namespace std;
using namespace cv;

void Matcher::LoadImage()
{
    test_image = cv::imread ( test_image_name, CV_LOAD_IMAGE_COLOR );
    dataset_image = cv::imread ( dataset_image_name, CV_LOAD_IMAGE_COLOR );
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1);
    cv::Mat distortion_coefficients = cv::Mat(4,1, CV_32FC1);
    camera_matrix.at<float>(0,0) = calibration.camera0_fu;
    camera_matrix.at<float>(0,1) = 0;
    camera_matrix.at<float>(0,2) = calibration.camera0_cu;
    camera_matrix.at<float>(1,0) = 0;
    camera_matrix.at<float>(1,1) = calibration.camera0_fv;
    camera_matrix.at<float>(1,2) = calibration.camera0_cv;
    camera_matrix.at<float>(2,0) = 0;
    camera_matrix.at<float>(2,1) = 0;
    camera_matrix.at<float>(2,2) = 1;
    distortion_coefficients.at<float>(0) = calibration.distortion_coefficients0[0];
    distortion_coefficients.at<float>(1) = calibration.distortion_coefficients0[1];
    distortion_coefficients.at<float>(2) = calibration.distortion_coefficients0[2];
    distortion_coefficients.at<float>(3) = calibration.distortion_coefficients0[3];
    cv::undistort(test_image, test_image_refined, camera_matrix, distortion_coefficients);
    cv:undistort(dataset_image, dataset_image_refined, camera_matrix, distortion_coefficients);
    cv::imwrite("../data/dataset_image_refined.png",dataset_image_refined);
    cv::imwrite("../data/test_image_refined.png",test_image_refined);
    // test_image_refined = test_image.clone();
    // dataset_image_refined = dataset_image.clone();
}

void Matcher::ORBMatch()
{
    
    Ptr<FeatureDetector> detector = ORB::create(40);
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    // Mat mask = Mat::zeros(test_image_refined.size(), CV_8U);  // type of mask is CV_8U
    // Mat roi0(mask, cv::Rect(0,0,test_image_refined.cols,test_image_refined.rows/10));
    // roi0 = Scalar(255); 
    Mat mask = Mat::zeros(test_image_refined.size(), CV_8U);  // type of mask is CV_8U
    Mat roi(mask, cv::Rect(0,0,test_image_refined.cols,test_image_refined.rows/10));//Rect四个形参分别是：x坐标，y坐标，长，高；注意(x,y)指的是矩形的左上角点

    roi = Scalar(255);
    detector->detect ( test_image_refined, keypoints_1, mask);
    detector->detect ( dataset_image_refined, keypoints_2, mask);
    for(int i = 1; i < 10; i++)
    {
        Mat mask_temp = Mat::zeros(test_image_refined.size(), CV_8U);  // type of mask is CV_8U
        int rows1 = i*test_image_refined.rows/10;
        int rows2 = test_image_refined.rows/10;
        cout << "rows1 = " << rows1 << "  rows2 = " << rows2 << endl;
        Mat roi_temp(mask_temp, cv::Rect(0, rows1, test_image_refined.cols, rows2));
        roi_temp = Scalar(255);
        vector<KeyPoint> keypoints_1_temp;
        vector<KeyPoint> keypoints_2_temp;
        detector->detect ( test_image_refined, keypoints_1_temp, mask_temp);
        detector->detect ( dataset_image_refined, keypoints_2_temp, mask_temp);
        for(int j = 0; j < keypoints_1_temp.size(); j++)
            keypoints_1.push_back(keypoints_1_temp.at(j));
        for(int j = 0; j < keypoints_2_temp.size(); j++)
            keypoints_2.push_back(keypoints_2_temp.at(j));
    }

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( test_image_refined, keypoints_1, descriptors_1 );
    descriptor->compute ( dataset_image_refined, keypoints_2, descriptors_2 );

    Mat outimg1;
    drawKeypoints( test_image_refined, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow("ORB feature",outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    // vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    
    // 仅供娱乐的写法
    // min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    // max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    // std::vector< DMatch > good_matches;
    
    // for (int i = 0; i < matches.size()/12; i ++)
    // for (int i = 0; i < 10; i ++)
    // {
    //     float min_distance = 0;
    //     int min_distance_id = 0;
    //     min_distance = matches[0].distance;
    //     min_distance_id = 0;
    //     float 
    //     for ( int j = 1; j < descriptors_1.rows; j++ )
    //     {
    //         if ( matches[j].distance < min_distance)
    //         {
    //             min_distance = matches[j].distance;
    //             min_distance_id = j;
                
    //         }
    //     }
    //     good_matches.push_back ( matches[min_distance_id] );
    //     matches.erase(matches.begin()+min_distance_id);
    // }
    int index1, index2;
    for (int i = 0; i < matches.size()/5; i ++)
    {
        float min_distance = 0;
        int min_distance_id = 0;
        min_distance = matches[0].distance;
        min_distance_id = 0;
        for ( int j = 1; j < descriptors_1.rows; j++ )
        {
            if ( matches[j].distance < min_distance)
            {
                min_distance = matches[j].distance;
                min_distance_id = j;
                
            }
        }
        good_matches.push_back ( matches[min_distance_id] );
        matches.erase(matches.begin()+min_distance_id);
        index1 = good_matches.at(i).queryIdx;
		index2 = good_matches.at(i).trainIdx;
        test_key_points.push_back(keypoints_1.at(index1).pt);
        dataset_key_points.push_back(keypoints_2.at(index2).pt);
        // cout << "test_key_points = " << test_key_points.at(i) << endl;
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    cout << "raw match point number = " << matches.size() << endl;
    cout << "good match point number = " << good_matches.size() << endl;
    drawMatches ( test_image_refined, keypoints_1, dataset_image_refined, keypoints_2, matches, img_match );
    drawMatches ( test_image_refined, keypoints_1, dataset_image_refined, keypoints_2, good_matches, img_goodmatch );
    // imshow ( "all match points", img_match );
    imshow ( "optimized match points", img_goodmatch );
    waitKey(0);
    // cvDestroyWindow("all match points");
    cvDestroyWindow("optimized match points");
    //-- 输出对应匹配点的图像坐标 -----------------------------------------------------------
    // int index1, index2;
	// for (int i = 0; i < good_matches.size(); i++)//将匹配的特征点坐标赋给point
	// {
	// 	index1 = good_matches.at(i).queryIdx;
	// 	index2 = good_matches.at(i).trainIdx;
    //     test_key_points.push_back(keypoints_1.at(index1).pt);
    //     dataset_key_points.push_back(keypoints_2.at(index2).pt);
    //     cout << "test_key_points = " << test_key_points.at(i) << endl;
	// }

}