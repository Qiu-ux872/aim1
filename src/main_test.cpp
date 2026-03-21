#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "PreProcess.hpp"
#include "PnPSolver.hpp"
#include "KalmanTracker.hpp"
#include "SerialPort.hpp"
#include "UdpLogger.hpp"

using namespace std;
using namespace cv;

void drawArmor(const vector<Armor>& armors, Mat& frame){
    for(const auto& armor : armors){
        circle(frame, armor.armor_center, 5, Scalar(0, 0, 255), -1);
        vector<Point> intPts;
        for (const auto& pt : armor.armor_pts) {
            intPts.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
        }
        polylines(frame, intPts, true, Scalar(0, 255, 0), 3);
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

    fs["dist_coeffs"] >> distCoeffs;
    if (distCoeffs.empty()) {
        cerr << "【警告】文件中没有有效的 dist_coeffs，将使用零向量！" << endl;
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
    cout << "相机内参矩阵加载成功！" << endl;
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

    string videoPath = "data/blue1.mp4";
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件: " << videoPath << endl;
        return -1;
    }
    cout << "视频文件打开成功" << endl;

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

    UdpLogger udpLogger("127.0.0.1", 9870);
    if (!udpLogger.isOpen()) {
        cerr << "UDP 初始化失败，将无法发送调试数据" << endl;
    }

    namedWindow("Armor Tracking", WINDOW_AUTOSIZE);

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

        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "视频播放完毕，退出循环" << endl;
            break;
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

        // 控制台输出调试信息
        if (pnpRes.isValid) {
            cout << "PnP解算距离: " << pnpRes.distance << " mm" << endl;
            cout << "重力补偿前yaw: " << pnpRes.yaw << "°, pitch: " << pnpRes.pitch << "°" << endl;
        }
        if (tracker.isInitialized()) {
            cout << "重力补偿后yaw: " << aim.yaw << "°, pitch: " << aim.pitch << "°" << endl;
        }

        // 串口发送
        if (serial.isOpen() && tracker.isInitialized()) {
            if (!serial.sendAimAngle(aim)) {
                cerr << "串口发送失败" << endl;
            }
        }

        // UDP 发送到 PlotJuggler
        if (udpLogger.isOpen()) {
            udpLogger.send(
                timeStamp,
                pnpRes.isValid ? pnpRes.distance : 0.0,
                pnpRes.isValid ? pnpRes.yaw : 0.0,
                pnpRes.isValid ? pnpRes.pitch : 0.0,
                pnpRes.isValid ? pnpRes.roll : 0.0,
                aim.yaw, aim.pitch,
                estPos.x, estPos.y, estPos.z,
                predPos.x, predPos.y, predPos.z
            );
        }

        // 绘制卡尔曼滤波点
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

        // 绘制卡尔曼估计的装甲板框（黄色）
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
            string text = format("X:%.1f Y:%.1f Z:%.1f", pnpRes.position.x, pnpRes.position.y, pnpRes.position.z);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
            text = format("Yaw:%.2f Pitch:%.2f", pnpRes.yaw, pnpRes.pitch);
            putText(frame, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
        } else {
            putText(frame, "No Target", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
        }

        string windowName = "Armor Tracking - FPS: " + to_string((int)fps);
        setWindowTitle("Armor Tracking", windowName);
        imshow("Armor Tracking", frame);
        char key = waitKey(30);
        if (key == 'q' || key == 'Q') break;
    }

    cap.release();
    serial.close();
    destroyAllWindows();
    return 0;
}