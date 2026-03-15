#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Param.hpp"
#include "Config.hpp"

using namespace cv;
using namespace std;

class PnPSolver {
public:
    PnPSolver();
    // 从 calibration.yml 加载相机参数
    bool loadCameraParams(const string& filename = "config/calibration.yml");
    // 直接设置相机参数
    void setCameraParams(const Mat& cameraMat, const Mat& distMat);
    // 对装甲板进行 PnP 解算
    PnPResult solveArmorPnP(const Armor& armor);
    // 对四个图像点进行 PnP 解算
    PnPResult solvePnP(const vector<Point2f>& imagePoints);

private:
    Mat cameraMatrix;
    Mat distCoeffs;
    vector<Point3f> armorPoints;  // 小装甲板三维点
    void calculateEulerAngles(PnPResult& result);
};

// 角度解算器（含重力补偿）
class AngleSolver {
public:
    AngleSolver();
    AimAngle calculateAimAngle(const PnPResult& pnpResult);

private:
    float bulletSpeed;
    float gravity;
    Point3f cameraOffset;  // 相机相对于枪口的偏移（毫米）
    float solvePitch(float dz, float dy, float v, float g);
};

#endif