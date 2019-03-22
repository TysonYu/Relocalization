//
// create by Tiezheng YU on 3/14/2019
//

#include <KeyPointsMaker.h>

using namespace std;
using namespace cv;

void KeyPointsMaker::GetCameraMatrix()
{
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
}

void KeyPointsMaker::LoadImage()
{
    cout << "Loading train set images ......" << endl;
    for ( int i=0; i<TrainSetName.timestamps.size(); i++ )
    {
        string path = "/home/icey/Desktop/project/camera_localization/data/mav0/cam0/data/"+TrainSetName.timestamps.at(i)+".png";
        // cout << path << endl;
        // Mat row_image = imread(path);
        // Mat refined_image;
        // cv::undistort(row_image, refined_image, camera_matrix, distortion_coefficients);
        images.push_back(imread(path));
    }
}

void KeyPointsMaker::LoadPointCloud()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr original_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/icey/Desktop/project/camera_localization/data/data.pcd", *original_cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file data.pcd \n");
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i = 0; i < original_cloud->points.size(); i++)
    {
        pcl::PointXYZRGB point;
        point.x = original_cloud->points[i].x;
        point.y = original_cloud->points[i].y;
        point.z = original_cloud->points[i].z;
        point.r = original_cloud->points[i].intensity*255;
        point.g = original_cloud->points[i].intensity*255;
        point.b = original_cloud->points[i].intensity*255;
        rgb_cloud->points.push_back(point);
    }
    world_cloud = *rgb_cloud;
}

void KeyPointsMaker::AddKeyPoints(int train_image_number)
{
    cout << "Adding train set KeyPoints ...... number: " << train_image_number << endl;
    Mat refined_image;
    cv::undistort(images.at(train_image_number), refined_image, camera_matrix, distortion_coefficients);
    Ptr<FeatureDetector> detector = ORB::create(80);
    vector<KeyPoint> keypoints;
    vector<KeyPoint> keypoints_temp;
    vector<Point2f> dataset_key_points;
    Mat mask = Mat::zeros(refined_image.size(), CV_8U);  // type of mask is CV_8U
    Mat roi(mask, cv::Rect(0,0,refined_image.cols,refined_image.rows/10));//Rect四个形参分别是：x坐标，y坐标，长，高；注意(x,y)指的是矩形的左上角点
    roi = Scalar(255);
    detector->detect ( refined_image, keypoints, mask);
    for(int i = 1; i < 10; i++)
    {
        Mat mask_temp = Mat::zeros(refined_image.size(), CV_8U);  // type of mask is CV_8U
        Mat roi_temp(mask_temp, cv::Rect(0, i*refined_image.rows/10, refined_image.cols, refined_image.rows/10));
        roi_temp = Scalar(255);
        detector->detect ( refined_image, keypoints_temp, mask_temp);
        for(int j = 0; j < keypoints_temp.size(); j++)
            keypoints.push_back(keypoints_temp.at(j));
    }
    
    //--------- 找到并把descriptor存入总的Descriptors中---------------------------------------
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    cout << "Finding descriptors ......" << endl;
    Mat descriptors;
    descriptor->compute ( refined_image, keypoints, descriptors);
    if(keypoints.size() == descriptors.rows)
        cout << "Descriptor all good ..." << endl;
    else
        cout << " There are bad Descriptor ..." << endl;
    train_descriptor.push_back(descriptors);
    for(int i = 0; i < keypoints.size(); i++)
    {
        train_keypoints.push_back(keypoints.at(i));
        dataset_key_points.push_back(keypoints.at(i).pt);
        train_key_points_2d.push_back(keypoints.at(i).pt);
    }
    // std::cout << train_descriptor << std::endl;
    //-------------------------------------------------------------------------------------
    // cout << "开始计算keypoints对应世界坐标系位置" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud = world_cloud.makeShared();//世界坐标系下的点云
    GroundTruth groundtruth(TrainSetName.timestamps.at(train_image_number));//得到GT的pose
    float p_RS_R_x = atof(groundtruth.groundtruth.at(0).c_str());
    float p_RS_R_y = atof(groundtruth.groundtruth.at(1).c_str());
    float p_RS_R_z = atof(groundtruth.groundtruth.at(2).c_str());
    float q_RS_w = atof(groundtruth.groundtruth.at(3).c_str());
    float q_RS_x = atof(groundtruth.groundtruth.at(4).c_str());
    float q_RS_y = atof(groundtruth.groundtruth.at(5).c_str());
    float q_RS_z = atof(groundtruth.groundtruth.at(6).c_str());
    float rt1 = 1 - 2*q_RS_y*q_RS_y - 2*q_RS_z*q_RS_z;
    float rt2 = 2*q_RS_x*q_RS_y - 2*q_RS_w*q_RS_z;
    float rt3 = 2*q_RS_x*q_RS_z + 2*q_RS_w*q_RS_y;
    float rt4 = p_RS_R_x;
    float rt5 = 2*q_RS_x*q_RS_y + 2*q_RS_w*q_RS_z;
    float rt6 = 1 - 2*q_RS_x*q_RS_x - 2*q_RS_z*q_RS_z;
    float rt7 = 2*q_RS_y*q_RS_z - 2*q_RS_w*q_RS_x;
    float rt8 = p_RS_R_y;
    float rt9 = 2*q_RS_x*q_RS_z - 2*q_RS_w*q_RS_y;
    float rt10 = 2*q_RS_y*q_RS_z + 2*q_RS_w*q_RS_x;
    float rt11 = 1 - 2*q_RS_x*q_RS_x - 2*q_RS_y*q_RS_y;
    float rt12 = p_RS_R_z;
    float rt13 = 0;
    float rt14 = 0;
    float rt15 = 0;
    float rt16 = 1;
    cv::Mat rt_temp2world = (cv::Mat_<float>(4, 4) << rt1 , rt2 , rt3 , rt4 , rt5 , rt6 , rt7 , rt8 , rt9 , rt10 , rt11 , rt12 , rt13 , rt14 , rt15 , rt16);
    cv::Mat rt_world2temp = rt_temp2world.inv();
    cv::Mat rt_temp2body = rt_body2temp.inv();
    cv::Mat rt_body2cam0 = rt_cam02body.inv();
    cv::Mat rt_world2cam0 = rt_body2cam0*rt_temp2body*rt_world2temp;
    // cv::Mat rt_world2cam0 = rt_body2cam0*rt_temp2body;
    // cout << "Rt matrix is : " << endl;
    // cout << rt_world2cam0 << endl;
    Eigen::Matrix4f rt;
    rt                    << rt_world2cam0.at<float>(0,0), rt_world2cam0.at<float>(0,1), rt_world2cam0.at<float>(0,2), rt_world2cam0.at<float>(0,3), 
                          rt_world2cam0.at<float>(1,0), rt_world2cam0.at<float>(1,1), rt_world2cam0.at<float>(1,2), rt_world2cam0.at<float>(1,3), 
                          rt_world2cam0.at<float>(2,0), rt_world2cam0.at<float>(2,1), rt_world2cam0.at<float>(2,2), rt_world2cam0.at<float>(2,3), 
                          rt_world2cam0.at<float>(3,0), rt_world2cam0.at<float>(3,1), rt_world2cam0.at<float>(3,2), rt_world2cam0.at<float>(3,3);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);//转化到相机坐标系
    pcl::transformPointCloud (*cloud, *camera_cloud, rt);//image，Z上未归一化的像素坐标系
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
    intrisic << calibration.camera0_fu, 0, calibration.camera0_cu, 0,    0, calibration.camera0_fv, calibration.camera0_cv, 0,    0, 0, 1, 0,    0, 0, 0, 1;
    pcl::transformPointCloud (*camera_cloud_filtered, *image_form_cloud, intrisic);//result，Z上未归一化的像素坐标系
    for(int i = 0; i < image_form_cloud->points.size(); i++)
    {
        image_form_cloud->points[i].x = image_form_cloud->points[i].x / image_form_cloud->points[i].z;
        image_form_cloud->points[i].y = image_form_cloud->points[i].y / image_form_cloud->points[i].z;
        if(image_form_cloud->points[i].z > max_depth)
            max_depth = image_form_cloud->points[i].z;
    }
    //--//把相机图片颜色投影到点云图上--测试用
    // cout << "把相机图片颜色投影到点云图上" << endl;
    // cv::Mat camera_image = cv::imread("../data/dataset_image_refined.png",cv::IMREAD_GRAYSCALE);
    // for(int i=0;i<image_form_cloud->points.size();i++)
    // {

    //     if(image_form_cloud->points[i].x>=0  && image_form_cloud->points[i].x<calibration.camera_width && image_form_cloud->points[i].y>=0 && image_form_cloud->points[i].y<calibration.camera_height)
    //     {
    //         camera_cloud_filtered->points[i].r = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
    //         camera_cloud_filtered->points[i].g = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
    //         camera_cloud_filtered->points[i].b = camera_image.at<uchar>(round(image_form_cloud->points[i].y),round(image_form_cloud->points[i].x));
    //     }
    // }
    //-- 把点云投影到图像上
    // cout << "把点云投影到图像上" << endl;
    cv::Mat M(calibration.camera_height, calibration.camera_width, CV_32F);//把点投影到M上
    cv::Mat P(calibration.camera_height, calibration.camera_width, CV_32F);//扩展投影点
    cv::MatIterator_<float>Mbegin,Mend;//遍历所有像素，初始化像素值
	for (Mbegin=M.begin<float>(),Mend=M.end<float>();Mbegin!=Mend;++Mbegin)
		*Mbegin=max_depth;
    for(int i=0;i<image_form_cloud->points.size();i++)//把深度值投影到图像M上
    {
        if(image_form_cloud->points[i].x>=0  && image_form_cloud->points[i].x<calibration.camera_width && image_form_cloud->points[i].y>=0 && image_form_cloud->points[i].y<calibration.camera_height)
        {
            if( camera_cloud_filtered->points[i].z < M.at<float>(image_form_cloud->points[i].y,image_form_cloud->points[i].x))
                M.at<float>(image_form_cloud->points[i].y,image_form_cloud->points[i].x) = camera_cloud_filtered->points[i].z;
        }
    }
    for(int count = 0; count < 5; count ++)
    {
        if (count%2 == 0) 
        {
            for (int i=1;i<M.rows-1;i++)
	        {
		        for (int j=1;j<M.cols-1;j++)
		        {
                    if(M.at<float>(i,j) == max_depth)
                    {
                        float temp = max_depth;
                        float sum = 0;
                        for(int n = i-1; n < i+2; n++)
                        {
                            for(int m = j-1; m < j+2; m++)
                            {
                                if(M.at<float>(n,m) < temp )
                                {
                                    temp = M.at<float>(n,m);
                                }   
                            }
                        }
                        P.at<float>(i,j) = temp;
                    }
                    else
                        P.at<float>(i,j)  = M.at<float>(i,j);
		        }
            }
        }
        else
        {
            for (int i=1;i<M.rows-1;i++)
	        {
		        for (int j=1;j<M.cols-1;j++)
		        {
                    if(P.at<float>(i,j) == max_depth)
                    {
                        float temp = max_depth;
                        for(int n = i-1; n < i+2; n++)
                        {
                            for(int m = j-1; m < j+2; m++)
                            {
                                if(P.at<float>(n,m) < temp)
                                {
                                    temp = P.at<float>(n,m);
                                }
                            }
                        }
                        M.at<float>(i,j) = temp;
                    }
                    else
                        M.at<float>(i,j)  = P.at<float>(i,j);
		        }
            }
        }
    }
    //---------　这一部分把dataset的keypoints投影到dataset 的图像上-------　测试用----
    // for (int i = 0; i < dataset_key_points.size(); i++)
    // {
    //     circle(refined_image, dataset_key_points.at(i), 1,cv::Scalar(255,0,0),-1);
    //     // cout << M.at<float>(dataset_key_points.at(i).y,dataset_key_points.at(i).x) << endl;
    // }
    // cv::imshow("depth map", refined_image);
    // cv::waitKey(0);
    // cvDestroyWindow("depth map");
    //---  获得keypoints在世界坐标系下的坐标---------------------
    // cout << "获得keypoints在世界坐标系下的坐标" << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // cout << endl << "max_depth = " << max_depth << endl;
    for (int i = 0;i < dataset_key_points.size(); i++)
    {
        pcl::PointXYZ point;
        float min_depth = max_depth;
        for(int m = -2; m < 2 ; m++)
            for(int n = -2; n < 2; n++)
            {
                if(min_depth > M.at<float>(dataset_key_points.at(i).y+m,dataset_key_points.at(i).x+n))
                    min_depth = M.at<float>(dataset_key_points.at(i).y+m,dataset_key_points.at(i).x+n);
            }
		point.z = min_depth;
        // cout << point.z << endl;
		point.x = dataset_key_points.at(i).x * point.z;
		point.y = dataset_key_points.at(i).y * point.z;
		keypoint_cloud->points.push_back(point);
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_cam0co_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f intrisic_inv = intrisic.inverse();
    pcl::transformPointCloud (*keypoint_cloud, *keypoint_cam0co_cloud, intrisic_inv);
    Eigen::Matrix4f rt_inv = rt.inverse();
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_world_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*keypoint_cam0co_cloud, *keypoint_world_cloud, rt_inv);
    //--------- 得到世界坐标系下的keypoints坐标------------------------------------------------------------
    // cout << "得到世界坐标系下的keypoints坐标" << endl;
    for(int i = 0; i < dataset_key_points.size(); i++)
    {
        cv::Point3f point;
        point.x = keypoint_world_cloud->points[i].x;
        point.y = keypoint_world_cloud->points[i].y;
        point.z = keypoint_world_cloud->points[i].z;
        train_key_points_3d.push_back(point);
        // cout << train_key_points_3d.at(i) << endl;
    }
}


void KeyPointsMaker::ShowKeyPointCloud()
{
    for(int i = 0; i < train_key_points_3d.size(); i++)
    {
        pcl::PointXYZRGB point;
        point.x = train_key_points_3d.at(i).x;
        point.y = train_key_points_3d.at(i).y;
        point.z = train_key_points_3d.at(i).z;
        point.r = 255;
        point.g = 255;
        point.b = 255;
        result_cloud.push_back(point);
    }
    // std::string filename("keypoints.pcd");
    // pcl::PCDWriter writer;
    // writer.write(filename,result_cloud);
    pcl::visualization::PCLVisualizer viewer("Keypoints viewer");
    viewer.addPointCloud(result_cloud.makeShared(), "sample cloud");
    viewer.setBackgroundColor(0,0,0);
    viewer.addCoordinateSystem();
    viewer.addCoordinateSystem();
    while(!viewer.wasStopped())
        viewer.spinOnce();
}