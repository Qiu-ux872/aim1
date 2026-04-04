#pragma once

#include <opencv2/opencv.hpp>
#include "Param.hpp"
#include "Config.hpp"
#include "EKF.hpp"
#include <memory>

class KalmanTracker{
public:
    KalmanTracker();

    void init(const cv::Point3f& position, double timeStamp);
    cv::Point3f predict(double timeStamp);
    cv::Point3f update(const cv::Point3f measuredPos, double timeStamp);
    cv::Point3f getEstimatedPosition() const;
    cv::Point3f getPredictionPosition() const { return m_predictedPose; }
    bool isInitialized() const { return m_initialized; }

    void initYaw(double yaw, double timeStamp);
    double predictYaw(double timeStamp);
    double updateYaw(double measuredYaw, double timeStamp);
    double getEstimatedYaw() const;
    double getPredictedYaw() const { return m_predictedYaw; }
    bool isYawInitialized() const { return m_yawInitialized; }

private:
    std::unique_ptr<EKF> m_kf;

    double m_lastTimePos;    // 位置滤波器时间戳
    double m_lastTimeYaw;    // Yaw滤波器时间戳
    double m_dt;

    cv::Point3f m_predictedPose;
    double m_predictedYaw;

    bool m_initialized;
    bool m_yawInitialized;

    void setTransitionMatrix(double dt);
    void loadParamsFromConfig();
};