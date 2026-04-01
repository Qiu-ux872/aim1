#pragma once

#include <opencv2/opencv.hpp>
#include "Param.hpp"
#include "Config.hpp"
#include "EKF.hpp"

class KalmanTracker{
public:
    KalmanTracker();

    // 位置跟踪接口
    void init(const cv::Point3f& position, double timeStamp);
    cv::Point3f predict(double timeStamp);
    cv::Point3f update(const cv::Point3f measuredPos, double timeStamp);
    cv::Point3f getEstimatedPosition() const;
    cv::Point3f getPredictionPosition() const { return m_predictedPose; }
    bool isInitialized() const { return m_initialized; }

    // Yaw 跟踪接口 (复用同一个 EKF 状态)
    void initYaw(double yaw, double timeStamp);
    double predictYaw(double timeStamp);
    double updateYaw(double measuredYaw, double timeStamp);
    double getEstimatedYaw() const;
    double getPredictedYaw() const { return m_predictedYaw; }
    bool isYawInitialized() const { return m_yawInitialized; }

private:
    // 统一的扩展卡尔曼滤波器: 8维状态 [x,y,z,vx,vy,vz,yaw,yaw_vel]
    std::unique_ptr<EKF> m_kf;

    // 时间管理
    double m_lastTime;
    double m_dt;

    // 预测和估计结果缓存
    cv::Point3f m_predictedPose;
    double m_predictedYaw;

    // 初始化标志
    bool m_initialized;
    bool m_yawInitialized;

    // 辅助方法
    void setTransitionMatrix(double dt);
    void loadParamsFromConfig();
};