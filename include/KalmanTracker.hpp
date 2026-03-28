#pragma once

#include <opencv2/opencv.hpp>
#include "Param.hpp"
#include "Config.hpp"
#include "EKF.hpp"   // 新增

class KalmanTracker{
public:
    KalmanTracker();

    // 原有接口保持不变
    void init(const cv::Point3f& position, double timeStamp);
    cv::Point3f predict(double timeStamp);
    cv::Point3f update(const cv::Point3f measuredPos, double timeStamp);
    cv::Point3f getEstimatedPosition() const;
    cv::Point3f getPredictionPosition() const { return m_predictedPose; }
    bool isInitialized() const { return m_initialized; }

    // 新增 yaw 滤波器接口（也可直接复用 EKF 状态）
    void initYaw(double yaw, double timeStamp);
    double predictYaw(double timeStamp);
    double updateYaw(double measuredYaw, double timeStamp);
    double getEstimatedYaw() const;
    double getPredictedYaw() const { return m_predictedYaw; }
    bool isYawInitialized() const { return m_yawInitialized; }

private:
    // 原位置滤波器（现在用 EKF 统一处理）
    std::unique_ptr<EKF> m_ekf;      // 8维状态 [x,y,z,vx,vy,vz,yaw,yaw_vel]
    double m_lastTime;
    double m_dt;
    cv::Point3f m_predictedPose;
    bool m_initialized;

    // yaw 相关（已被 EKF 包含，但保留独立接口以便兼容）
    double m_lastYawTime;
    double m_predictedYaw;
    bool m_yawInitialized;

    void setTransitionMatrix(double dt);
    void loadParamInConfig();
    void loadYawParamsFromConfig();
};