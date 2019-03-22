//
// Greated by Tiezheng YU on 3/15/2019
//

#include <PNPSolver.h>

using namespace std;
using namespace cv;


void PNPSolver::FindMatches(int test_image_number)
{
    cv::Mat test_image_refined;
    //--------- 计算得到keypoints--------------------------------------------------
    std::cout << " imput test image ......" << std::endl;
    Ptr<FeatureDetector> test_detector = ORB::create(80);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    Mat test_image = imread("/home/icey/Desktop/project/camera_localization/data/V1_02_medium/mav0/cam0/data/" + TestImageName.timestamps.at(test_image_number) + ".png");
    cv::undistort(test_image, test_image_refined, KeyPoints.camera_matrix, KeyPoints.distortion_coefficients);
    cv::imwrite("../data/test_image_refined.png",test_image_refined);
    //--------- 得到测试集的keypoints ---------------------------------------------
    std::cout << "get test image keypoints ......" << std::endl;
    vector<KeyPoint> keypoints;
    vector<KeyPoint> keypoints_temp;
    vector<Point2f> test_key_points_2d;
    Mat mask = Mat::zeros(test_image_refined.size(), CV_8U);  // type of mask is CV_8U
    Mat roi(mask, cv::Rect(0,0,test_image_refined.cols,test_image_refined.rows/10));//Rect四个形参分别是：x坐标，y坐标，长，高；注意(x,y)指的是矩形的左上角点
    roi = Scalar(255);
    test_detector->detect ( test_image_refined, keypoints, mask);
    for(int i = 1; i < 10; i++)
    {
        Mat mask_temp = Mat::zeros(test_image_refined.size(), CV_8U);  // type of mask is CV_8U
        int rows1 = i*test_image_refined.rows/10;
        int rows2 = test_image_refined.rows/10;
        Mat roi_temp(mask_temp, cv::Rect(0, rows1, test_image_refined.cols, rows2));
        roi_temp = Scalar(255);
        vector<KeyPoint> keypoints_1_temp;
        test_detector->detect ( test_image_refined, keypoints_1_temp, mask_temp);
        for(int j = 0; j < keypoints_1_temp.size(); j++)
        {
            keypoints.push_back(keypoints_1_temp.at(j));
        }
    }
    
    //--------- 计算得到descriptor ------------------------------------------------
    std::cout << "find matches between test image and train images ...... " << std::endl; 
    Ptr<DescriptorExtractor> test_descriptor = ORB::create();
    Mat descriptors;
    test_descriptor->compute ( test_image_refined, keypoints, descriptors);
    // for (int i = 0; i < keypoints.size(); i++)
    //     cout << keypoints.at(i).pt << endl;
    cout << "test image descriptor number = " << descriptors.rows << endl;
    std::vector<cv::DMatch> matches;
    matcher->match ( descriptors, KeyPoints.train_descriptor, matches );
    cout << "raw match number = " << matches.size() << endl;
    //--------- 选择好的match -----------------------------------------------------
    std::cout << "get good matches from row matches ...... " << std::endl;
    std::vector<cv::DMatch > good_matches;
    std::vector<cv::Point2f> good_train_2d_points;
    std::vector<cv::Point2f> good_test_2d_points;
    std::vector<cv::Point3f> good_train_3d_points;
    int index1, index2;
    for (int i = 0; i < matches.size()/2; i ++)
    // for (int i = 0; i < 10; i ++)
    {
        float min_distance = 0;
        int min_distance_id = 0;
        min_distance = matches[0].distance;
        min_distance_id = 0;
        for ( int j = 1; j < descriptors.rows; j++ )
        {
            if ( matches[j].distance < min_distance)
            {
                min_distance = matches[j].distance;
                min_distance_id = j;
                
            }
        }
        // cout << matches[min_distance_id].distance << endl;
        good_matches.push_back ( matches[min_distance_id] );
        matches.erase(matches.begin()+min_distance_id);
        index1 = good_matches.at(i).queryIdx;
		index2 = good_matches.at(i).trainIdx;
        // index1 = matches.at(i).queryIdx;
		// index2 = matches.at(i).trainIdx;
        good_test_2d_points.push_back(keypoints.at(index1).pt);
        good_train_3d_points.push_back(KeyPoints.train_key_points_3d.at(index2));

    }
    cout << "dataset descriptor number = " << KeyPoints.train_descriptor.rows << endl;
    cout << "dataset keypoint number = " << KeyPoints.train_keypoints.size() << endl;
    cout << " good match number = " << good_matches.size() << endl;
    // for (int i = 0; i < good_test_2d_points.size(); i++)
    // {
    //     circle(test_image_refined, good_test_2d_points.at(i), 3,cv::Scalar(255,0,0),0);
    // }
    // cv::imshow("depth map", test_image_refined);
    // cv::waitKey(0);
    // cvDestroyWindow("depth map");
    cout << "using PnpRANSAC to get the extrinsic matrix .........." << endl;
    cv::Mat distCoeffs = (cv::Mat_<float>(4,1) << 0, 0, 0, 0);
    cv::Mat rvec = (cv::Mat_<float>(3,1) << 0, 0, 0);
    cv::Mat tvec = (cv::Mat_<float>(3,1) << 0, 0, 0);
    bool result = true;
    cv::solvePnPRansac(good_train_3d_points, good_test_2d_points, KeyPoints.camera_matrix, distCoeffs, rvec, tvec);
    cv::Mat rotation = (cv::Mat_<float>(3,3) << 0,0,0, 0,0,0, 0,0,0);
    cv::Rodrigues(rvec,rotation);
    // cout << rotation << endl;
    R = rotation;
    T = tvec;
    cout << " R = " << endl << R << endl;
    cout << " T = " << endl << T << endl;

    pcl::PointCloud<pcl::PointXYZRGB> result_cloud;
    for(int i = 0; i < good_train_3d_points.size(); i++)
    {
        pcl::PointXYZRGB point;
        point.x = good_train_3d_points.at(i).x;
        point.y = good_train_3d_points.at(i).y;
        point.z = good_train_3d_points.at(i).z;
        point.r = 255;
        point.g = 255;
        point.b = 255;
        result_cloud.push_back(point);
    }
    std::string filename("keypoints.pcd");
    pcl::PCDWriter writer;
    writer.write(filename,result_cloud);
    // pcl::visualization::PCLVisualizer viewer("result viewer");
    // viewer.addPointCloud(result_cloud.makeShared(), "sample cloud");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"sample cloud");//设置点的某些属性，比如点的大小
    // viewer.setBackgroundColor(0,0,0);
    // viewer.addCoordinateSystem();
    // viewer.addCoordinateSystem();
    // while(!viewer.wasStopped())
    //     viewer.spinOnce();
}



void PNPSolver::ShowResults()
{
    Eigen::Matrix4f rt_world2cam;
    rt_world2cam << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), T.at<double>(0),
                    R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), T.at<double>(1),
                    R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), T.at<double>(2),
                    0, 0, 0, 1.0;
    // std::cout << "R = " << R.at<double>(0,0) << std::endl;
    // std::cout << "T = " << T << std::endl;
    // cout << rt_world2cam << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud = KeyPoints.world_cloud.makeShared();//世界坐标系下的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);//转化到相机坐标系
    pcl::transformPointCloud (*cloud, *camera_cloud, rt_world2cam);//image，Z上未归一化的像素坐标系
    //-- //滤掉后面的点
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (camera_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 50);//delete all the point that z<0 && z>40
    pass.filter (*camera_cloud_filtered);
    
    //--  图像形式点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr image_form_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    Eigen::Matrix4f intrisic;//相机内参
    intrisic << KeyPoints.calibration.camera0_fu, 0, KeyPoints.calibration.camera0_cu, 0,    0, KeyPoints.calibration.camera0_fv, KeyPoints.calibration.camera0_cv, 0,    0, 0, 1, 0,    0, 0, 0, 1;
    pcl::transformPointCloud (*camera_cloud_filtered, *image_form_cloud, intrisic);//result，Z上未归一化的像素坐标系
    for(int i = 0; i < image_form_cloud->points.size(); i++)
    {
        image_form_cloud->points[i].x = image_form_cloud->points[i].x / image_form_cloud->points[i].z;
        image_form_cloud->points[i].y = image_form_cloud->points[i].y / image_form_cloud->points[i].z;
        if(image_form_cloud->points[i].z > KeyPoints.max_depth)
            KeyPoints.max_depth = image_form_cloud->points[i].z;
    }
    //--//把相机图片颜色投影到点云图上
    // cvCvtColor(test_image_refined, camera_image, CV_BGR2GRAY);
    cv::Mat camera_image = cv::imread("../data/test_image_refined.png",cv::IMREAD_GRAYSCALE);
    for(int i=0;i<image_form_cloud->points.size();i++)
    {

        if(image_form_cloud->points[i].x>=0  && image_form_cloud->points[i].x<KeyPoints.calibration.camera_width && image_form_cloud->points[i].y>=0 && image_form_cloud->points[i].y<KeyPoints.calibration.camera_height)
        {
            camera_cloud_filtered->points[i].r = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
            camera_cloud_filtered->points[i].g = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
            camera_cloud_filtered->points[i].b = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
        }
    }
    result_cloud = * camera_cloud_filtered;
    // waitKey(0);
    // pcl::visualization::PCLVisualizer viewer("Result viewer");
    // viewer.addPointCloud(camera_cloud_filtered, "sample cloud");
    // viewer.setBackgroundColor(0,0,0);
    // viewer.addCoordinateSystem();
    // viewer.addCoordinateSystem();
    // while(!viewer.wasStopped())
    //     // while (!viewer->wasStopped ())
    //     viewer.spinOnce();
}