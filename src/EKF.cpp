#include "EKF.hpp"
#include "Config.hpp"
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

    // 观测矩阵 H (4x8)
    m_H.at<double>(0,0) = 1.0;  // x
    m_H.at<double>(1,1) = 1.0;  // y
    m_H.at<double>(2,2) = 1.0;  // z
    m_H.at<double>(3,6) = 1.0;  // yaw
}

void ExtendedKalmanFilter::loadParamsFromConfig() {
    const auto& c = Config::get().kalman;

    // 过程噪声 Q (8x8)
    m_Q.at<double>(0,0) = c.processNoisePos;
    m_Q.at<double>(1,1) = c.processNoisePos;
    m_Q.at<double>(2,2) = c.processNoisePos;
    m_Q.at<double>(3,3) = c.processNoiseVel;
    m_Q.at<double>(4,4) = c.processNoiseVel;
    m_Q.at<double>(5,5) = c.processNoiseVel;
    m_Q.at<double>(6,6) = c.yawProcessNoisePos;
    m_Q.at<double>(7,7) = c.yawProcessNoiseVel;

    // 观测噪声 R (4x4)
    m_R.at<double>(0,0) = c.measurementNoisePos;
    m_R.at<double>(1,1) = c.measurementNoisePos;
    m_R.at<double>(2,2) = c.measurementNoisePos;
    m_R.at<double>(3,3) = c.yawMeasurementNoise;

    // 初始协方差 P
    m_P = cv::Mat::eye(8, 8, CV_64F) * c.initialErrorCov;

    std::cout << "[EKF] 初始化完成，参数: "
              << "posNoise=" << c.processNoisePos
              << ", velNoise=" << c.processNoiseVel
              << ", measNoise=" << c.measurementNoisePos
              << ", yawPosNoise=" << c.yawProcessNoisePos
              << ", yawVelNoise=" << c.yawProcessNoiseVel
              << ", yawMeasNoise=" << c.yawMeasurementNoise << std::endl;
}

void ExtendedKalmanFilter::setTransitionMatrix(double dt) {
    // 状态转移矩阵 F (8x8) 线性模型
    cv::setIdentity(m_F);
    m_F.at<double>(0,3) = dt;  // x += vx*dt
    m_F.at<double>(1,4) = dt;  // y += vy*dt
    m_F.at<double>(2,5) = dt;  // z += vz*dt
    m_F.at<double>(6,7) = dt;  // yaw += yaw_vel*dt
}

cv::Mat ExtendedKalmanFilter::stateTransition(const cv::Mat& x, double dt) {
    cv::Mat x_new = x.clone();
    x_new.at<double>(0) = x.at<double>(0) + x.at<double>(3) * dt;
    x_new.at<double>(1) = x.at<double>(1) + x.at<double>(4) * dt;
    x_new.at<double>(2) = x.at<double>(2) + x.at<double>(5) * dt;
    x_new.at<double>(6) = x.at<double>(6) + x.at<double>(7) * dt;
    return x_new;
}

cv::Mat ExtendedKalmanFilter::stateJacobian(double dt) {
    cv::Mat J = cv::Mat::eye(8, 8, CV_64F);
    J.at<double>(0,3) = dt;
    J.at<double>(1,4) = dt;
    J.at<double>(2,5) = dt;
    J.at<double>(6,7) = dt;
    return J;
}

cv::Mat ExtendedKalmanFilter::measurementModel(const cv::Mat& x) {
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = x.at<double>(0);
    z.at<double>(1) = x.at<double>(1);
    z.at<double>(2) = x.at<double>(2);
    z.at<double>(3) = x.at<double>(6);
    return z;
}

cv::Mat ExtendedKalmanFilter::measurementJacobian() {
    // 观测矩阵 H 是常数
    return m_H;
}

void ExtendedKalmanFilter::predictState(double dt) {
    // 状态预测
    m_x = stateTransition(m_x, dt);
    // 协方差预测
    cv::Mat J = stateJacobian(dt);
    m_P = J * m_P * J.t() + m_Q;
}

void ExtendedKalmanFilter::correct(const cv::Mat& z) {
    // 卡尔曼增益
    cv::Mat S = m_H * m_P * m_H.t() + m_R;
    cv::Mat K = m_P * m_H.t() * S.inv();

    // 新息
    cv::Mat z_pred = measurementModel(m_x);
    cv::Mat y = z - z_pred;

    // 状态更新
    m_x = m_x + K * y;

    // 协方差更新 (Joseph form)
    m_P = (m_I - K * m_H) * m_P * (m_I - K * m_H).t() + K * m_R * K.t();
}

void ExtendedKalmanFilter::init(const cv::Point3f& position, double yaw, double timeStamp) {
    m_x.at<double>(0) = position.x;
    m_x.at<double>(1) = position.y;
    m_x.at<double>(2) = position.z;
    m_x.at<double>(3) = 0.0;  // vx
    m_x.at<double>(4) = 0.0;  // vy
    m_x.at<double>(5) = 0.0;  // vz
    m_x.at<double>(6) = yaw;
    m_x.at<double>(7) = 0.0;  // yaw_vel

    m_lastTimePos = timeStamp;
    m_lastTimeYaw = timeStamp;
    m_predictedPose = position;
    m_predictedYaw = yaw;
    m_initialized = true;
    m_yawInitialized = true;
}

void ExtendedKalmanFilter::predict(double timeStamp) {
    if (!m_initialized) return;

    // 计算 dt（使用位置时间戳，因为两者共用）
    double dt = timeStamp - m_lastTimePos;
    if (dt > 0.1) dt = 0.033;
    if (dt < 0.001) dt = 0.033;

    setTransitionMatrix(dt);
    predictState(dt);

    // 提取预测的位置和 yaw
    m_predictedPose.x = m_x.at<double>(0);
    m_predictedPose.y = m_x.at<double>(1);
    m_predictedPose.z = m_x.at<double>(2);
    m_predictedYaw = m_x.at<double>(6);

    m_lastTimePos = timeStamp;
    m_lastTimeYaw = timeStamp;
}

void ExtendedKalmanFilter::updatePosition(const cv::Point3f& measuredPos, double timeStamp) {
    if (!m_initialized) {
        init(measuredPos, 0.0, timeStamp);
        return;
    }

    // 先预测到当前时间
    predict(timeStamp);

    // 构造观测向量 (位置部分)
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = measuredPos.x;
    z.at<double>(1) = measuredPos.y;
    z.at<double>(2) = measuredPos.z;
    z.at<double>(3) = m_x.at<double>(6);  // yaw 保持原值

    correct(z);
}

void ExtendedKalmanFilter::updateYaw(double measuredYaw, double timeStamp) {
    if (!m_yawInitialized) {
        // 若尚未初始化 yaw，但位置可能已初始化，则仅设置 yaw
        m_x.at<double>(6) = measuredYaw;
        m_x.at<double>(7) = 0.0;
        m_lastTimeYaw = timeStamp;
        m_predictedYaw = measuredYaw;
        m_yawInitialized = true;
        return;
    }

    // 预测到当前时间（使用 yaw 时间戳）
    double dt = timeStamp - m_lastTimeYaw;
    if (dt > 0.1) dt = 0.033;
    if (dt < 0.001) dt = 0.033;
    setTransitionMatrix(dt);
    predictState(dt);

    // 构造观测向量 (yaw 部分)
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = m_x.at<double>(0);  // 位置不变
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