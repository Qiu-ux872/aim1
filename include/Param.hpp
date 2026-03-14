#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct LightBar{
    Point2f bar_center;         // 灯条中心点
    vector<Point2f> bar_pts;    // 灯条角点
    float bar_width;            // 灯条宽度
    float bar_length;           // 灯条长度
    float bar_angle;            // 灯条偏转角
};

struct Armor{
    Point2f armor_center;       // 装甲板中心点
    LightBar left;              // 左灯条
    LightBar right;             // 右灯条
    vector<Point2f> armor_pts;  // 装甲板角点
    float armor_angle;          // 装甲板角度
    float armor_width;          // 装甲板宽度
    float armor_height;         // 装甲板高度
    float yaw;                  // 装甲板yaw轴
    float pitch;                // 装甲板pitch轴
};

struct PnPResult{
    Point3f position;           // 平移向量（x, y, z）单位：mm
    Mat rvec;                   // 旋转向量(3x1)
    Mat rMatrix;                // 旋转矩阵(3x3)
    Mat tvec;                   // 平移矩阵(3x1)
    double distance;            // 距离(单位：mm)
    float yaw;                  // yaw轴(rad)
    float pitch;                // pitch轴(rad)
    float roll;                 // roll轴(rad)
    bool isValid;               // 解算是否有效
};