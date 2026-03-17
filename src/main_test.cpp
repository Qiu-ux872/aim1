#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "PreProcess.hpp"
#include "PnPSolver.hpp"
#include "KalmanTracker.hpp"
#include "SerialPort.hpp"

using namespace std;
using namespace cv;


void drawArmor(const vector<Armor>& armors, Mat& frame){
    for(const auto& armor : armors){
        // 绘制灯条中心（红色圆点）
        circle(frame, armor.armor_center, 5, Scalar(0, 0, 255), -1);
        
        // 将浮点角点转换为整数点
        vector<Point> intPts;
        for (const auto& pt : armor.armor_pts) {
            intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
        }
        // 使用 polylines 绘制装甲板轮廓
        polylines(frame, intPts, true, Scalar(0, 255, 0), 3);
    }
}

// 获取当前时间戳（s）——用于卡尔曼滤波，视频调试时可基于帧计数模拟时间
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

int main() {
    // 1. 加载全局配置
    Config::get();
    cout << "配置加载成功" << endl;

    // 2. 加载相机内参矩阵用于投影
    Mat cameraMatrix;
    loadCameraMatrix("config/calibration.yml", cameraMatrix);

    // 3. 打开视频文件（请根据实际文件名修改路径）
    string videoPath = "data/red1.mp4";  // 请替换为您的视频文件名
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件: " << videoPath << endl;
        return -1;
    }
    cout << "视频文件打开成功" << endl;

    // 4. 初始化 PnP 解算器（其内部会加载相机参数用于解算）
    PnPSolver pnpSolver;
    if (!pnpSolver.loadCameraParams("config/calibration.yml")) {
        cerr << "警告：PnP解算器使用默认相机内参" << endl;
    }

    // 5. 创建卡尔曼跟踪器和角度解算器
    KalmanTracker tracker;
    AngleSolver angleSolver;   // 构造函数已从 Config 读取弹道参数

    // 6. 初始化串口（如需发送角度）
    SerialPort serial;
    if (!serial.open()) {
        cerr << "串口打开失败，将无法发送角度" << endl;
    }

    // 7. 创建显示窗口
    namedWindow("Armor Tracking", WINDOW_AUTOSIZE);

    // 8. 主循环
    double lastTime = getCurrentTimeSec();
    int frameCount = 0;
    double fps = 0.0;

    while (true) {
        double timeStamp = getCurrentTimeSec();
        frameCount++;

        // 每秒更新一次 fps 变量
        if (timeStamp - lastTime >= 1.0) {
            fps = frameCount / (timeStamp - lastTime);
            frameCount = 0;
            lastTime = timeStamp;
        }

        // 从视频读取一帧
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "视频播放完毕，退出循环" << endl;
            break;
        }

        // 预处理：获得二值图
        Mat blur = PreProcess::process(frame);

        // 检测灯条
        vector<LightBar> lightBars = PreProcess::detectLightBars(blur);

        // 匹配装甲板
        vector<Armor> armors = PreProcess::detectArmors(lightBars);

        // 绘制装甲板
        drawArmor(armors, frame);

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

                // 绘制装甲板框（绿色线条）
                const auto& pts = target.armor_pts;
                for (int i = 0; i < 4; i++) {
                    line(frame, pts[i], pts[(i+1)%4], Scalar(0, 255, 0), 2);
                }
            }
        }

        // 卡尔曼滤波更新/预测
        Point3f estPos, predPos;
        if (hasTarget) {
            if (!tracker.isInitialized()) {
                tracker.init(measuredPos, timeStamp);
                estPos = measuredPos;
                predPos = measuredPos;
            } else {
                estPos = tracker.update(measuredPos, timeStamp);
                predPos = tracker.getPredictionPosition();  // 注意方法名：getPredictionPosition()
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
            // 使用估计位置作为目标位置计算角度
            // 构造一个临时 PnPResult 仅包含位置信息
            PnPResult dummy;
            dummy.position = predPos;  // 预测的下一时刻位置
            dummy.distance = norm(estPos);  // 粗略距离，实际 AngleSolver 只使用 position
            aim = angleSolver.calculateAimAngle(dummy);
        }

        // Debug
        if (pnpRes.isValid) {
            cout << "PnP解算距离: " << pnpRes.distance << " mm" << endl;
            cout << "重力补偿前yaw: " << pnpRes.yaw << "°, pitch: " << pnpRes.pitch << "°" << endl;
        }
        if (tracker.isInitialized()) {
            cout << "重力补偿后yaw: " << aim.yaw << "°, pitch: " << aim.pitch << "°" << endl;
        }

        // 通过串口发送瞄准角度
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

        // 绘制卡尔曼估计的装甲板框（黄色）
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

    // 转换为整数点并绘制闭合多边形
    vector<Point> intPts;
    for (const auto& pt : imgPts) {
        intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
    }
    polylines(frame, intPts, true, Scalar(0, 255, 255), 2); // 黄色线条
}

        // 显示图像
        string windowName = "Armor Tracking - FPS: " + to_string((int)fps);
        setWindowTitle("Armor Tracking", windowName);
        // 输出目标信息
        // string tvecText = "Pos: x " + to_string(static_cast<int>(measuredPos.x)) + " y " + to_string(static_cast<int>(measuredPos.y)) + " z " + to_string(static_cast<int>(measuredPos.z));
        // putText(frame, tvecText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        Config& c = Config::get();
        imshow("Armor Tracking", frame);
        char key = waitKey(100);  // 等待1ms，若需要按帧率播放可适当增加
        if (key == 'q' || key == 'Q') break;
        
    }

    // 9. 清理资源
    cap.release();
    serial.close();
    destroyAllWindows();

    return 0;
}