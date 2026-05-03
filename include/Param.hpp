#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// 灯条结构体
struct LightBar{
    Point2f bar_center;      // 灯条中心点坐标
    vector<Point2f> bar_pts;   // 灯角点
    float bar_width;         // 灯条宽度
    float bar_length;        // 灯条长度
    float bar_angle;         // 灯条角度
};

// 装甲板结构体
struct Armor{
    Point2f armor_center;    // 装甲板中心点
    LightBar left;           // 装甲板左侧灯条
    LightBar right;         // 装甲板右侧灯条
    vector<Point2f> armor_pts;  // 装甲板角点
    float armor_angle;       // 装甲板角度
    float armor_width;       // 装甲板宽度
    float armor_height;      // 装甲板高度
    double distance_mm = 0.0;  // 装甲板到相机的实际距离
};

// PnP解算结果结构体
struct PnPResult {
    Point3f position = Point3f(0,0,0);  // 目标三维坐标（相机坐标系下）
    Mat rotationVec;    // 旋转向量（3×1）
    Mat rotationMatrix; // 旋转矩阵（3×3）
    Mat translationVec; // 平移向量（3×1）
    double distance = 0.0;     // 目标距离
    double yaw = 0.0;           // 偏航角
    double pitch = 0.0;         // 俯仰角
    double roll = 0.0;          // 横滚角
    double filteredYaw = 0.0;  // 滤波后的偏航角
    double predictedYaw = 0.0; // 预测的偏航角
    bool isValid = false;       // 解算结果是否有效
    double reprojectionError = 0.0;  // 重投影误差
};

// 打击角度结构体
struct AimAngle {
    float yaw;      // 偏航角
    float pitch;    // 俯仰角
    float distance; // 预测距离 用于弹道补偿
};
