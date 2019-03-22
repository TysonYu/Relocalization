//
// Greated by Tiezheng YU on 3/12/2019
// 

#include <LoopDetector.h>

using namespace std;
using namespace cv;

void LoopDetector::LoadVocabulary()
{

    // read the images and database  
    cout<<"reading database"<<endl;
    DBoW3::Vocabulary temp("./vocabulary.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want: 
    if ( temp.empty() )
    {
        cerr<<"Vocabulary does not exist."<<endl;
    }
    LoopDetector::vocab = temp;
}

void LoopDetector::LoadImage()
{
    cout<<"reading images... "<<endl;
    for ( int i=0; i<names.timestamps.size(); i++ )
    {
        string path = "/home/icey/Desktop/project/camera_localization/data/mav0/cam0/data/"+names.timestamps.at(i)+".png";
        // if(i%6 == 0)
            images.push_back( imread(path) );
    }
}

void LoopDetector::CreateFeatures()
{
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    // Ptr< Feature2D > detector = ORB::create();
    for ( Mat& image:images )//找出训练集的所有descriptors
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
}

int LoopDetector::FindID(std::string test_name)
{
    Mat descriptor;//找出待查询图像的descriptor
    vector<KeyPoint> keypoints;
    Mat test = imread(test_name);
    detector->detectAndCompute( test, Mat(), keypoints, descriptor );
    DBoW3::BowVector v1;
    vocab.transform( descriptor, v1 );
    cout<<"comparing images with images "<<endl;
    int max_id = 0;
    double max_score = 0.0;
    for ( int j=0; j<images.size(); j++ )
    {
        DBoW3::BowVector v2;
        vocab.transform( descriptors[j], v2 );
        double score = vocab.score(v1, v2);
        if (score > max_score)
        {
            max_score = score;
            max_id = j;
        }
        // cout<<"test "<<" vs image "<<j<<" : "<<score<<endl;
    }
    cout << "best similar image = " << names.timestamps.at(max_id) << endl;
    cout << "best similar image number = " << max_id << endl;
    id_finded = max_id;
    return id_finded;
}
