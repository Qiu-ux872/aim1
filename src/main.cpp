#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "CameraDriver.hpp"
#include "PreProcess.hpp"
#include "PnPSolver.hpp"
#include "KalmanTracker.hpp"
#include "SerialPort.hpp"

using namespace std;
using namespace cv;

// 绘制检测到的装甲板（白色框）
void drawArmor(const vector<Armor>& armors, Mat& frame){
    for(const auto& armor : armors){
        // 绘制装甲板中心（红色圆点）
        circle(frame, armor.armor_center, 5, Scalar(0, 0, 255), -1);
        
        // 将浮点角点转换为整数点
        vector<Point> intPts;
        for (const auto& pt : armor.armor_pts) {
            intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
        }
        // 使用 polylines 绘制装甲板轮廓（白色）
        polylines(frame, intPts, true, Scalar(255, 255, 255), 3);
    }
}

// 获取当前时间戳（s）
double getCurrentTimeSec() {
    auto now = chrono::steady_clock::now();
    return chrono::duration<double>(now.time_since_epoch()).count();
}

// 从 calibration.yml 加载相机内参矩阵
bool loadCameraMatrix(const string& filename, Mat& cameraMatrix) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "【警告】无法打开相机参数文件: " << filename << "，将使用默认内参！" << endl;
        cameraMatrix = (Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs.release();

    if (cameraMatrix.empty() || cameraMatrix.rows != 3 || cameraMatrix.cols != 3) {
        cerr << "【警告】文件中没有有效的 camera_matrix，将使用默认内参！" << endl;
        cameraMatrix = (Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        return false;
    }

    cout << "相机内参矩阵加载成功！" << endl;
    return true;
}

// 将三维点投影到图像平面
Point2f projectPoint(const Point3f& pt, const Mat& cameraMatrix) {
    vector<Point3f> pts3d = {pt};
    vector<Point2f> pts2d;
    cv::projectPoints(pts3d, Mat::zeros(3,1,CV_64F), Mat::zeros(3,1,CV_64F),
                      cameraMatrix, noArray(), pts2d);
    return pts2d[0];
}

// 绘制灯条（可选）
void drawLightBar(const vector<LightBar>& detected_bars, Mat& frame){
    for(const auto& bar : detected_bars){
        // 绘制灯条中心（红色圆点）
        circle(frame, bar.bar_center, 5, Scalar(0, 0, 255), -1);
        
        // 绘制旋转矩形
        const auto& pts = bar.bar_pts;
        for (int i = 0; i < 4; i++) {
            line(frame, pts[i], pts[(i+1)%4], Scalar(0, 255, 0), 2);
        }
    }
}

int main() {
    // 1. 加载全局配置
    Config::get();
    cout << "配置加载成功" << endl;

    // 2. 加载相机内参矩阵用于投影
    Mat cameraMatrix;
    loadCameraMatrix("config/calibration.yml", cameraMatrix);

    // 3. 初始化相机
    CameraDriver camera;
    if (!camera.open()) {
        cerr << "相机打开失败" << endl;
        return -1;
    }
    if (!camera.start()) {
        cerr << "相机启动失败" << endl;
        return -1;
    }

    // 4. 初始化 PnP 解算器（其内部会再次加载相机参数用于解算）
    PnPSolver pnpSolver;
    if (!pnpSolver.loadCameraParams("config/calibration.yml")) {
        cerr << "警告：PnP解算器使用默认相机内参" << endl;
    }

    // 5. 创建卡尔曼跟踪器和角度解算器
    KalmanTracker tracker;
    AngleSolver angleSolver;   // 构造函数已从 Config 读取弹道参数

    // 6. 初始化串口
    SerialPort serial;
    if (!serial.open()) {
        cerr << "串口打开失败，将无法发送角度" << endl;
    }

    // 7. 创建显示窗口
    namedWindow("Armor Tracking", WINDOW_AUTOSIZE);

    // 8. 主循环
    while (true) {
        double timeStamp = getCurrentTimeSec();

        // 捕获一帧图像
        Mat frame = camera.capture(1000);
        if (frame.empty()) {
            cerr << "捕获图像超时，继续等待..." << endl;
            continue;
        }

        // 预处理：获得二值图
        Mat binary = PreProcess::process(frame);

        // 检测灯条
        vector<LightBar> lightBars = PreProcess::detectLightBars(binary);

        // 匹配装甲板
        vector<Armor> armors = PreProcess::detectArmors(lightBars);

        // 绘制灯条（可选）
        drawLightBar(lightBars, frame);

        // 默认初始化为无效状态
        bool hasTarget = false;
        Point3f measuredPos;
        PnPResult pnpRes;

        if (!armors.empty()) {
            // 简单选取第一个装甲板
            const Armor& target = armors[0];

            // PnP 解算
            pnpRes = pnpSolver.solveArmorPnP(target);
            if (pnpRes.isValid) {
                hasTarget = true;
                measuredPos = pnpRes.position;

                // 绘制检测到的装甲板框（白色，已由 drawArmor 完成，但此处也可保留备用）
                // 但 drawArmor 会在后面统一绘制，所以这里可以不重复绘制
            }
        }

        // 绘制检测到的装甲板（白色）
        drawArmor(armors, frame);

        // 卡尔曼滤波更新/预测
        Point3f estPos, predPos;
        if (hasTarget) {
            if (!tracker.isInitialized()) {
                tracker.init(measuredPos, timeStamp);
                estPos = measuredPos;
                predPos = measuredPos;
            } else {
                estPos = tracker.update(measuredPos, timeStamp);
                predPos = tracker.getPredictionPosition();
            }
        } else {
            if (tracker.isInitialized()) {
                predPos = tracker.predict(timeStamp);
                estPos = tracker.getEstimatedPosition();
            }
        }

        // 计算瞄准角度
        AimAngle aim;
        if (tracker.isInitialized()) {
            PnPResult dummy;
            dummy.position = estPos;  // 使用当前估计位置计算角度
            dummy.distance = norm(estPos);
            aim = angleSolver.calculateAimAngle(dummy);
        }

        // 通过串口发送瞄准角度（如果串口已打开）
        if (serial.isOpen() && tracker.isInitialized()) {
            if (!serial.sendAimAngle(aim)) {
                cerr << "串口发送失败" << endl;
            }
        }

        // 绘制卡尔曼滤波点（需要投影到图像平面）
        if (tracker.isInitialized()) {
            // 实测点（蓝色）
            if (hasTarget) {
                Point2f ptMeas = projectPoint(measuredPos, cameraMatrix);
                circle(frame, ptMeas, 5, Scalar(255, 0, 0), -1);
            }
            // 估计点（白色）
            Point2f ptEst = projectPoint(estPos, cameraMatrix);
            circle(frame, ptEst, 5, Scalar(255, 255, 255), -1);
            // 预测点（红色）
            Point2f ptPred = projectPoint(predPos, cameraMatrix);
            circle(frame, ptPred, 5, Scalar(0, 0, 255), -1);
        }

        // 绘制卡尔曼估计的装甲板框（绿色）
        if (tracker.isInitialized()) {
            const float armorWidth = 135.0f;   // 小装甲板宽度（mm）
            const float armorHeight = 125.0f;  // 小装甲板高度（mm）

            // 在相机坐标系下，假设装甲板平面垂直于Z轴，计算四个角点
            vector<Point3f> objPts(4);
            objPts[0] = estPos + Point3f(-armorWidth/2, -armorHeight/2, 0); // 左上
            objPts[1] = estPos + Point3f( armorWidth/2, -armorHeight/2, 0); // 右上
            objPts[2] = estPos + Point3f( armorWidth/2,  armorHeight/2, 0); // 右下
            objPts[3] = estPos + Point3f(-armorWidth/2,  armorHeight/2, 0); // 左下

            // 投影到图像平面
            vector<Point2f> imgPts;
            for (const auto& pt : objPts) {
                imgPts.push_back(projectPoint(pt, cameraMatrix));
            }

            // 转换为整数点并绘制闭合多边形（绿色）
            vector<Point> intPts;
            for (const auto& pt : imgPts) {
                intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
            }
            polylines(frame, intPts, true, Scalar(0, 255, 0), 2);
        }

        // 显示图像
        imshow("Armor Tracking", frame);
        char key = waitKey(1);
        if (key == 'q' || key == 'Q') break;
    }

    // 9. 清理资源
    camera.stop();
    camera.close();
    serial.close();
    destroyAllWindows();

    return 0;
}