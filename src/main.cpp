#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "EKF.hpp"
#include "Config.hpp"
#include "CameraDriver.hpp"
#include "PreProcess.hpp"
#include "PnPSolver.hpp"
#include "KalmanTracker.hpp"
#include "SerialPort.hpp"
#include "plotter.hpp"   // 替换 UdpLogger.hpp

using namespace std;
using namespace cv;

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

double getCurrentTimeSec() {
    auto now = chrono::steady_clock::now();
    return chrono::duration<double>(now.time_since_epoch()).count();
}

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

    KalmanTracker tracker;
    AngleSolver angleSolver;

    SerialPort serial;
    if (!serial.open()) {
        cerr << "串口打开失败，将无法发送角度" << endl;
    }

    // UDP 日志（使用 Plotter）
    unique_ptr<tools::Plotter> plotter;
    if (Config::get().udp.enabled) {
        plotter = make_unique<tools::Plotter>(Config::get().udp.host, Config::get().udp.port);
        // Plotter 没有 isOpen() 方法，默认认为创建成功
        cout << "Plotter 已启用，目标 " << Config::get().udp.host << ":" << Config::get().udp.port << endl;
    } else {
        cout << "Plotter 发送已禁用" << endl;
    }

    namedWindow("Armor Tracking", WINDOW_NORMAL);

    double lastTime = getCurrentTimeSec();
    int frameCount = 0;
    double fps = 0.0;

    Point3f predPos(0, 0, 0);

    while (true) {
        double timeStamp = getCurrentTimeSec();
        frameCount++;

        if (timeStamp - lastTime >= 1.0) {
            fps = frameCount / (timeStamp - lastTime);
            frameCount = 0;
            lastTime = timeStamp;
        }

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

        bool hasTarget = false;
        Point3f measuredPos;
        PnPResult pnpRes;

        if (!armors.empty()) {
            const Armor& target = armors[0];
            pnpRes = pnpSolver.solveArmorPnP(target);
            if (pnpRes.isValid) {
                hasTarget = true;
                measuredPos = pnpRes.position;

                double filteredYaw = tracker.updateYaw(pnpRes.yaw, timeStamp);
                double predictedYaw = tracker.predictYaw(timeStamp);

                const auto& pts = target.armor_pts;
                for (int i = 0; i < 4; i++) {
                    line(frame, pts[i], pts[(i+1)%4], Scalar(0, 255, 0), 2);
                }
            }
        }

        Point3f estPos;
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

        AimAngle aim;
        if (tracker.isInitialized()) {
            PnPResult dummy;
            dummy.position = predPos;
            dummy.distance = norm(estPos);
            aim = angleSolver.calculateAimAngle(dummy);
        }

        if (pnpRes.isValid) {
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

        // 使用 Plotter 发送 UDP 数据
        if (plotter) {
            nlohmann::json j;
            j["timestamp"] = timeStamp;
            j["pnp_distance"] = pnpRes.isValid ? pnpRes.distance : 0.0;
            j["pnp_yaw"] = pnpRes.isValid ? pnpRes.yaw : 0.0;
            j["pnp_pitch"] = pnpRes.isValid ? pnpRes.pitch : 0.0;
            j["pnp_roll"] = pnpRes.isValid ? pnpRes.roll : 0.0;
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

        if (tracker.isInitialized()) {
            if (hasTarget) {
                Point2f ptMeas = projectPoint(measuredPos, cameraMatrix, distCoeffs);
                circle(frame, ptMeas, 5, Scalar(255, 0, 0), -1);
            }
            Point2f ptEst = projectPoint(estPos, cameraMatrix, distCoeffs);
            circle(frame, ptEst, 5, Scalar(255, 255, 255), -1);
            Point2f ptPred = projectPoint(predPos, cameraMatrix, distCoeffs);
            circle(frame, ptPred, 5, Scalar(0, 0, 255), -1);
        }

        if (tracker.isInitialized()) {
            const float armorWidth = 135.0f;
            const float armorHeight = 125.0f;

            vector<Point3f> objPts(4);
            objPts[0] = estPos + Point3f(-armorWidth/2, -armorHeight/2, 0);
            objPts[1] = estPos + Point3f( armorWidth/2, -armorHeight/2, 0);
            objPts[2] = estPos + Point3f( armorWidth/2,  armorHeight/2, 0);
            objPts[3] = estPos + Point3f(-armorWidth/2,  armorHeight/2, 0);

            vector<Point2f> imgPts;
            for (const auto& pt : objPts) {
                imgPts.push_back(projectPoint(pt, cameraMatrix, distCoeffs));
            }

            vector<Point> intPts;
            for (const auto& pt : imgPts) {
                intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
            }
            polylines(frame, intPts, true, Scalar(0, 255, 255), 2);
        }

        if (pnpRes.isValid) {
            string posText = format("X:%.1f Y:%.1f Z:%.1f", pnpRes.position.x, pnpRes.position.y, pnpRes.position.z);
            putText(frame, posText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
            string angleText = format("Yaw: %.2f (raw) / %.2f (filt) / %.2f (pred)", pnpRes.yaw, pnpRes.filteredYaw, pnpRes.predictedYaw);
            putText(frame, angleText, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
            string angleText2 = format("Pitch:%.2f Roll:%.2f", pnpRes.pitch, pnpRes.roll);
            putText(frame, angleText2, Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
        } else {
            putText(frame, "No Target", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
        }

        string windowName = "Armor Tracking - FPS: " + to_string((int)fps);
        setWindowTitle("Armor Tracking", windowName);
        imshow("Armor Tracking", frame);
        char key = waitKey(1);
        if (key == 'q' || key == 'Q') break;
    }

    camera.stop();
    camera.close();
    serial.close();
    destroyAllWindows();
    return 0;
}