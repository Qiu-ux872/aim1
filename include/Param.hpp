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
};

struct PnPResult {
    Point3f position;          // 平移向量 (x, y, z) 单位：毫米
    Mat rotationVec;           // 旋转向量 (3x1)
    Mat rotationMatrix;        // 旋转矩阵 (3x3)
    Mat translationVec;        // 平移矩阵 (3x1)
    double distance;               // 距离（毫米）
    double yaw;                    // yaw（弧度）
    double pitch;                  // pitch（弧度）
    double roll;                   // roll（弧度）
    bool isValid;                  // 是否有效
    double reprojectionError;      // 重投影误差
};

// 瞄准角度
struct AimAngle {
    float yaw;                     // 水平转角（弧度）
    float pitch;                   // 俯仰角（弧度）
    float distance;                // 距离（米）
    float flyTime;                 // 飞行时间（秒）
};