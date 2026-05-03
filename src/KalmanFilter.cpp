#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>

ExtendedKalmanFilter::ExtendedKalmanFilter()
    : m_x(cv::Mat::zeros(8, 1, CV_64F)),
      m_P(cv::Mat::eye(8, 8, CV_64F)),
      m_F(cv::Mat::eye(8, 8, CV_64F)),
      m_H(cv::Mat::zeros(4, 8, CV_64F)),
      m_Q(cv::Mat::eye(8, 8, CV_64F) * 1e-4),
      m_R(cv::Mat::eye(4, 4, CV_64F) * 1e-2),
      m_I(cv::Mat::eye(8, 8, CV_64F)),
      m_lastTimePos(0),
      m_lastTimeYaw(0),
      m_dt(0),
      m_predictedPose(0,0,0),
      m_predictedYaw(0),
      m_initialized(false),
      m_yawInitialized(false)
{
    loadParamsFromConfig();

    // 观测矩阵 H: 观测 [x, y, z, yaw]
    m_H.at<double>(0,0) = 1.0;
    m_H.at<double>(1,1) = 1.0;
    m_H.at<double>(2,2) = 1.0;
    m_H.at<double>(3,6) = 1.0;
}

void ExtendedKalmanFilter::loadParamsFromConfig() {
    const auto& cfg = Config::get().kalman;

    // 过程噪声 Q (8x8)
    m_Q.at<double>(0,0) = cfg.processNoisePos;
    m_Q.at<double>(1,1) = cfg.processNoisePos;
    m_Q.at<double>(2,2) = cfg.processNoisePos;
    m_Q.at<double>(3,3) = cfg.processNoiseVel;
    m_Q.at<double>(4,4) = cfg.processNoiseVel;
    m_Q.at<double>(5,5) = cfg.processNoiseVel;
    m_Q.at<double>(6,6) = cfg.yawProcessNoisePos;
    m_Q.at<double>(7,7) = cfg.yawProcessNoiseVel;

    // 观测噪声 R (4x4)
    m_R.at<double>(0,0) = cfg.measurementNoisePos;
    m_R.at<double>(1,1) = cfg.measurementNoisePos;
    m_R.at<double>(2,2) = cfg.measurementNoisePos;
    m_R.at<double>(3,3) = cfg.yawMeasurementNoise;

    // 初始协方差 P
    m_P = cv::Mat::eye(8, 8, CV_64F) * cfg.initialErrorCov;

    std::cout << "[EKF] 初始化完成，参数: "
              << "posNoise=" << cfg.processNoisePos
              << ", velNoise=" << cfg.processNoiseVel
              << ", measNoise=" << cfg.measurementNoisePos
              << ", yawPosNoise=" << cfg.yawProcessNoisePos
              << ", yawVelNoise=" << cfg.yawProcessNoiseVel
              << ", yawMeasNoise=" << cfg.yawMeasurementNoise << std::endl;
}

void ExtendedKalmanFilter::setTransitionMatrix(double dt) {
    cv::setIdentity(m_F);
    m_F.at<double>(0,3) = dt;   // x += vx * dt
    m_F.at<double>(1,4) = dt;   // y += vy * dt
    m_F.at<double>(2,5) = dt;   // z += vz * dt
    m_F.at<double>(6,7) = dt;   // yaw += yaw_vel * dt
}

void ExtendedKalmanFilter::predictState(double dt) {
    // 状态预测 x = F * x
    m_x = m_F * m_x;
    // 协方差预测 P = F * P * F^T + Q
    m_P = m_F * m_P * m_F.t() + m_Q;
}

void ExtendedKalmanFilter::correct(const cv::Mat& z) {
    // 卡尔曼增益 K = P * H^T * (H * P * H^T + R)^-1
    cv::Mat S = m_H * m_P * m_H.t() + m_R;
    cv::Mat K = m_P * m_H.t() * S.inv();

    // 新息 y = z - H * x
    cv::Mat y = z - m_H * m_x;

    // 状态更新
    m_x = m_x + K * y;

    // 协方差更新 (Joseph form)
    m_P = (m_I - K * m_H) * m_P * (m_I - K * m_H).t() + K * m_R * K.t();
}

void ExtendedKalmanFilter::init(const cv::Point3f& position, double yaw, double timeStamp) {
    m_x.at<double>(0) = position.x;
    m_x.at<double>(1) = position.y;
    m_x.at<double>(2) = position.z;
    m_x.at<double>(3) = 0.0;   // vx
    m_x.at<double>(4) = 0.0;   // vy
    m_x.at<double>(5) = 0.0;   // vz
    m_x.at<double>(6) = yaw;
    m_x.at<double>(7) = 0.0;   // yaw_vel

    m_lastTimePos = timeStamp;
    m_lastTimeYaw = timeStamp;
    m_predictedPose = position;
    m_predictedYaw = yaw;
    m_initialized = true;
    m_yawInitialized = true;
}

void ExtendedKalmanFilter::predict(double timeStamp) {
    if (!m_initialized) return;

    // 计算时间差 dt（使用位置时间戳）
    double dt = timeStamp - m_lastTimePos;
    if (dt > 0.1) dt = 0.033;
    if (dt < 0.001) dt = 0.033;

    setTransitionMatrix(dt);
    predictState(dt);

    m_predictedPose.x = m_x.at<double>(0);
    m_predictedPose.y = m_x.at<double>(1);
    m_predictedPose.z = m_x.at<double>(2);
    m_predictedYaw = m_x.at<double>(6);

    m_lastTimePos = timeStamp;
    m_lastTimeYaw = timeStamp;  // 保持同步
}

void ExtendedKalmanFilter::updatePosition(const cv::Point3f& measuredPos, double timeStamp) {
    if (!m_initialized) {
        init(measuredPos, 0.0, timeStamp);
        return;
    }

    // 先预测到当前时刻
    predict(timeStamp);

    // 构造观测向量（只更新位置，yaw 保持当前估计值）
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = measuredPos.x;
    z.at<double>(1) = measuredPos.y;
    z.at<double>(2) = measuredPos.z;
    z.at<double>(3) = m_x.at<double>(6);   // yaw 不变

    correct(z);
}

void ExtendedKalmanFilter::updateYaw(double measuredYaw, double timeStamp) {
    if (!m_yawInitialized) {
        // 仅设置 yaw，位置保持当前值
        m_x.at<double>(6) = measuredYaw;
        m_x.at<double>(7) = 0.0;
        m_lastTimeYaw = timeStamp;
        m_predictedYaw = measuredYaw;
        m_yawInitialized = true;
        return;
    }

    // 先预测到当前时刻（使用 yaw 时间差）
    double dt = timeStamp - m_lastTimeYaw;
    if (dt > 0.1) dt = 0.033;
    if (dt < 0.001) dt = 0.033;
    setTransitionMatrix(dt);
    predictState(dt);

    // 构造观测向量（只更新 yaw，位置保持不变）
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = m_x.at<double>(0);
    z.at<double>(1) = m_x.at<double>(1);
    z.at<double>(2) = m_x.at<double>(2);
    z.at<double>(3) = measuredYaw;

    correct(z);

    m_lastTimeYaw = timeStamp;
}

cv::Point3f ExtendedKalmanFilter::getEstimatedPosition() const {
    if (!m_initialized) return cv::Point3f(0,0,0);
    return cv::Point3f(m_x.at<double>(0), m_x.at<double>(1), m_x.at<double>(2));
}

double ExtendedKalmanFilter::getEstimatedYaw() const {
    if (!m_yawInitialized) return 0.0;
    return m_x.at<double>(6);
}
