#pragma once

#include <opencv2/opencv.hpp>
#include "Param.hpp"
#include "Config.hpp"
        
using namespace std;
using namespace cv;

class KalmanTracker{
public:
    KalmanTracker();

    // 原位置滤波器接口
    void init(const Point3f& position, double timeStamp);
    Point3f predict(double timeStamp);
    Point3f update(const Point3f measuredPos, double timeStamp);
    Point3f getEstimatedPosition() const;
    Point3f getPredictionPosition() const { return m_predictedPose; }
    bool isInitialized() const { return m_initialized; }

    // ========== 新增：yaw 滤波器接口 ==========
    void initYaw(double yaw, double timeStamp);
    double predictYaw(double timeStamp);
    double updateYaw(double measuredYaw, double timeStamp);
    double getEstimatedYaw() const;
    double getPredictedYaw() const { return m_predictedYaw; }
    bool isYawInitialized() const { return m_yawInitialized; }
    // =======================================

private:
    // 位置滤波器成员
    KalmanFilter m_kf;          // 6维状态，3维观测
    Mat m_state;
    Mat m_measured;
    double m_lastTime;
    double m_dt;
    Point3f m_predictedPose;
    bool m_initialized;

    void setTransitionMatrix(double dt);
    void loadParamInConfig();

    KalmanFilter m_kfYaw;
    double m_lastYawTime;
    double m_dtYaw;
    double m_predictedYaw;
    bool m_yawInitialized;

    void setYawTransitionMatrix(double dt);
    void loadYawParamsFromConfig();
};