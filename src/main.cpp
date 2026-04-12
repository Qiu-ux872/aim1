#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "Config.hpp"
#include "CameraDriver.hpp"
#include "PreProcess.hpp"
#include "PnPSolver.hpp"
#include "EKF.hpp"
#include "SerialPort.hpp"
#include "plotter.hpp"
#include "TargetSelect.hpp"

using namespace std;
using namespace cv;

// 辅助函数 绘制装甲板
void drawArmor(const vector<Armor>& armors, Mat& frame){
    for(const auto& armor : armors){
        circle(frame, armor.armor_center, 5, Scalar(0, 0, 255), -1);
        vector<Point> intPts;
        for (const auto& pt : armor.armor_pts) {
            intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
        }
        polylines(frame, intPts, true, Scalar(0, 255, 0), 1);
    }
}

// 获取当前时间戳
double getCurrentTimeSec() {
    auto now = chrono::steady_clock::now();
    return chrono::duration<double>(now.time_since_epoch()).count();
}

// 加载相机参数
bool loadCameraParams(const string& filename, Mat& cameraMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "【警告】无法打开相机参数文件: " << filename << "，将使用默认内参！" << endl;
        cameraMatrix = (Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        distCoeffs = Mat::zeros(5, 1, CV_64F);
        return false;
    }
    fs["camera_matrix"] >> cameraMatrix;
    if (cameraMatrix.empty() || cameraMatrix.rows != 3 || cameraMatrix.cols != 3) {
        cerr << "【警告】文件中没有有效的 camera_matrix，将使用默认内参！" << endl;
        cameraMatrix = (Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        distCoeffs = Mat::zeros(5, 1, CV_64F);
        fs.release();
        return false;
    }
    fs["distortion_coeffs"] >> distCoeffs;
    if (distCoeffs.empty()) {
        cerr << "【警告】文件中没有有效的 distortion_coeffs，将使用零向量！" << endl;
        distCoeffs = Mat::zeros(5, 1, CV_64F);
    } else {
        if (distCoeffs.rows == 1 && distCoeffs.cols > 1) {
            distCoeffs = distCoeffs.reshape(1, distCoeffs.cols);
        } else if (distCoeffs.cols != 1 || distCoeffs.rows < 4) {
            cerr << "【警告】畸变系数格式异常，将使用零向量！" << endl;
            distCoeffs = Mat::zeros(5, 1, CV_64F);
        }
    }
    PreProcess::camera_matrix = cameraMatrix.clone();
    PreProcess::dist_coeffs = distCoeffs.clone();
    cout << "相机参数加载成功！" << endl;
    return true;
}

// 3D点投影到2D
Point2f projectPoint(const Point3f& pt, const Mat& cameraMatrix, const Mat& distCoeffs) {
    vector<Point3f> pts3d = {pt};
    vector<Point2f> pts2d;
    cv::projectPoints(pts3d, Mat::zeros(3,1,CV_64F), Mat::zeros(3,1,CV_64F),
                      cameraMatrix, distCoeffs, pts2d);
    return pts2d[0];
}

int main() {
    Config::get();
    cout << "配置加载成功" << endl;

    Mat cameraMatrix, distCoeffs;
    loadCameraParams("config/calibration.yml", cameraMatrix, distCoeffs);

    CameraDriver camera;
    if (!camera.open()) {
        cerr << "相机打开失败" << endl;
        return -1;
    }
    if (!camera.start()) {
        cerr << "相机启动失败" << endl;
        return -1;
    }

    PnPSolver pnpSolver;
    if (!pnpSolver.loadCameraParams("config/calibration.yml")) {
        cerr << "警告：PnP解算器使用默认相机内参" << endl;
    }

    ExtendedKalmanFilter tracker;   // 目标跟踪器
    AngleSolver angleSolver;        // 角度解算器
    TargetSelector targetSelector(Config::get());   // 目标选择器

    SerialPort serial;
    if (!serial.open()) {
        cerr << "串口打开失败，将无法发送角度" << endl;
    }

    unique_ptr<tools::Plotter> plotter;
    if (Config::get().udp.enabled) {
        plotter = make_unique<tools::Plotter>(Config::get().udp.host, Config::get().udp.port);
        cout << "Plotter 已启用，目标 " << Config::get().udp.host << ":" << Config::get().udp.port << endl;
    } else {
        cout << "Plotter 发送已禁用" << endl;
    }

    namedWindow("Armor Tracking", WINDOW_NORMAL);

    double lastTime = getCurrentTimeSec();  // 用于计算FPS
    int frameCount = 0;                     // 预测位置和预测yaw
    double fps = 0.0;                       // 当前FPS
    Point3f predPos(0,0,0);                 // 预测位置
    double predictedYaw = 0.0;              // 预测yaw

    while (true) {
        double timeStamp = getCurrentTimeSec();
        frameCount++;
        if (timeStamp - lastTime >= 1.0) {
            fps = frameCount / (timeStamp - lastTime);
            frameCount = 0;
            lastTime = timeStamp;
        }

        // 捕获图像
        Mat frame = camera.capture(1000);
        if (frame.empty()) {
            cerr << "捕获图像超时，继续等待..." << endl;
            continue;
        }

        Mat binary = PreProcess::process(frame);
        vector<LightBar> lightBars = PreProcess::detectLightBars(binary);
        vector<Armor> armors;
        if (tracker.isInitialized()) {
            armors = PreProcess::detectArmors(lightBars, &predPos);
        } else {
            armors = PreProcess::detectArmors(lightBars, nullptr);
        }

        drawArmor(armors, frame);

        // 为每个装甲板进行 PnP 解算并填充距离
        for (auto& armor : armors) {
            PnPResult res = pnpSolver.solveArmorPnP(armor);
            if (res.isValid) {
                armor.distance_mm = res.distance;
            } else {
                armor.distance_mm = 0.0;
            }
        }

        bool hasTarget = false;
        Point3f measuredPos;
        PnPResult pnpRes{};
        double filteredYaw = 0.0;

        if (!armors.empty()) {
            double timestamp_ms = timeStamp * 1000.0;
            const Armor* best = targetSelector.select(armors, timestamp_ms);
            if (best) {
                pnpRes = pnpSolver.solveArmorPnP(*best);
                if (pnpRes.isValid) {
                    hasTarget = true;
                    measuredPos = pnpRes.position;

                    // 更新卡尔曼滤波器
                    if (!tracker.isInitialized()) {
                        tracker.init(measuredPos, pnpRes.yaw, timeStamp);
                    } else {
                        tracker.updatePosition(measuredPos, timeStamp);
                        tracker.updateYaw(pnpRes.yaw, timeStamp);
                    }
                    filteredYaw = tracker.getEstimatedYaw();
                    predictedYaw = tracker.getPredictedYaw();

                    const auto& pts = best->armor_pts;
                    for (int i = 0; i < 4; i++) {
                        line(frame, pts[i], pts[(i+1)%4], Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        // 获取估计位置和预测位置
        Point3f estPos = tracker.getEstimatedPosition();
        if (tracker.isInitialized()) {
            if (!hasTarget) {
                // 没有观测时，仅预测
                tracker.predict(timeStamp);
                predPos = tracker.getPredictedPosition();
                predictedYaw = tracker.getPredictedYaw();
                estPos = tracker.getEstimatedPosition();
            } else {
                predPos = tracker.getPredictedPosition();
            }
        }

        AimAngle aim;
        if (tracker.isInitialized()) {
            PnPResult dummy;        // 构造一个临时的 PnPResult 结构体，填充位置和 yaw 用于角度解算
            dummy.position = predPos;
            dummy.distance = norm(estPos);
            aim = angleSolver.calculateAimAngle(dummy);
        }

        if (hasTarget) {
            cout << "PnP解算距离: " << pnpRes.distance << " mm" << endl;
            cout << "重力补偿前yaw: " << pnpRes.yaw << "度, pitch: " << pnpRes.pitch << "度" << endl;
        }
        if (tracker.isInitialized()) {
            cout << "重力补偿后yaw: " << aim.yaw << "度, pitch: " << aim.pitch << "度" << endl;
        }

        if (serial.isOpen() && tracker.isInitialized()) {
            if (!serial.sendAimAngle(aim)) {
                cerr << "串口发送失败" << endl;
            }
        }

        if (plotter) {
            nlohmann::json j;
            j["timestamp"] = timeStamp;
            if (hasTarget) {
                j["pnp_distance"] = pnpRes.distance;
                j["pnp_yaw"] = pnpRes.yaw;
                j["pnp_pitch"] = pnpRes.pitch;
                j["pnp_roll"] = pnpRes.roll;
            } else {
                j["pnp_distance"] = 0.0;
                j["pnp_yaw"] = 0.0;
                j["pnp_pitch"] = 0.0;
                j["pnp_roll"] = 0.0;
            }
            j["aim_yaw"] = aim.yaw;
            j["aim_pitch"] = aim.pitch;
            j["est_x"] = estPos.x;
            j["est_y"] = estPos.y;
            j["est_z"] = estPos.z;
            j["pred_x"] = predPos.x;
            j["pred_y"] = predPos.y;
            j["pred_z"] = predPos.z;
            plotter->plot(j);
        }

        // 绘制卡尔曼滤波点
        if (tracker.isInitialized()) {
            if (hasTarget) {
                Point2f ptMeas = projectPoint(measuredPos, cameraMatrix, distCoeffs);
                circle(frame, ptMeas, 3, Scalar(255, 0, 0), -1);
            }
            Point2f ptEst = projectPoint(estPos, cameraMatrix, distCoeffs);
            circle(frame, ptEst, 3, Scalar(255, 255, 255), -1);
            Point2f ptPred = projectPoint(predPos, cameraMatrix, distCoeffs);
            circle(frame, ptPred, 3, Scalar(0, 0, 255), -1);
        }

        // 绘制卡尔曼预测的装甲板框（黄色）
        if (tracker.isInitialized()) {
            const float armorWidth = 135.0f;
            const float armorHeight = 125.0f;
            vector<Point3f> objPts(4);
            objPts[0] = predPos + Point3f(-armorWidth/2, -armorHeight/2, 0);
            objPts[1] = predPos + Point3f( armorWidth/2, -armorHeight/2, 0);
            objPts[2] = predPos + Point3f( armorWidth/2,  armorHeight/2, 0);
            objPts[3] = predPos + Point3f(-armorWidth/2,  armorHeight/2, 0);
            vector<Point2f> imgPts;     // 投影到图像平面上的点
            for (const auto& pt : objPts) imgPts.push_back(projectPoint(pt, cameraMatrix, distCoeffs));
            vector<Point> intPts;   // 整数像素点用于绘制
            for (const auto& pt : imgPts) intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
            polylines(frame, intPts, true, Scalar(0, 255, 255), 2);
        }

        if (hasTarget) {
            string posText = format("X:%.1f Y:%.1f Z:%.1f", pnpRes.position.x, pnpRes.position.y, pnpRes.position.z);
            putText(frame, posText, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
            string angleText = format("Yaw: %.2f (raw) / %.2f (filt) / %.2f (pred)", pnpRes.yaw, filteredYaw, predictedYaw);
            putText(frame, angleText, Point(10,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
            string angleText2 = format("Pitch:%.2f Roll:%.2f", pnpRes.pitch, pnpRes.roll);
            putText(frame, angleText2, Point(10,70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
        } else {
            putText(frame, "No Target", Point(10,30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
        }

        string windowName = "Armor Tracking - FPS: " + to_string((int)fps);
        setWindowTitle("Armor Tracking", windowName);
        imshow("Armor Tracking", frame);
        if (waitKey(1) == 'q') break;
    }

    camera.stop();
    camera.close();
    serial.close();
    destroyAllWindows();
    return 0;
}