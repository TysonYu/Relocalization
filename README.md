# location a camera in a given point cloud

#### 项目概述
+ 通过一个点云以及一组训练集的图像和groung truth(需要手动纠正初始位姿)，获得一个从所有图像中获得的keypoints cloud。这里的每一个keypoint对应一个descriptor同时我们通过匹配每一幅测试集图像中的keypoints来得到一个对应。选出有用的keypoints。然后获得相机位姿。
+ 输出测试集图像在点云中的位置。(中间过程可选择输出由keypoints组成的点云)。


#### 依赖

+ OpenCV 3.3.1
+ libpcl
+ Eigen
+ cmake

#### 构建

1. mkdir build
2. cd build
3. cmake -DCMAKE_BUILD_TYPE=Release ..
4. make -j4
5. ./main
