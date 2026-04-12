#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct LightBar{
    Point2f bar_center;
    vector<Point2f> bar_pts;
    float bar_width;
    float bar_length;
    float bar_angle;
};

struct Armor{
    Point2f armor_center;
    LightBar left;
    LightBar right;
    vector<Point2f> armor_pts;
    float armor_angle;
    float armor_width;
    float armor_height;
    double distance_mm = 0.0;
};

struct PnPResult {
    Point3f position = Point3f(0,0,0);
    Mat rotationVec;
    Mat rotationMatrix;
    Mat translationVec;
    double distance = 0.0;
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;
    double filteredYaw = 0.0;
    double predictedYaw = 0.0;
    bool isValid = false;
    double reprojectionError = 0.0;
};

struct AimAngle {
    float yaw;
    float pitch;
    float distance;
    float flyTime;
};