#include "KalmanTracker.hpp"
#include <cmath>

KalmanTracker::KalmanTracker()
    : m_ekf(new EKF(8, 4)),    // 8维状态，4维观测
      m_lastTime(0),
      m_dt(0),
      m_initialized(false),
      m_lastYawTime(0),
      m_predictedYaw(0),
      m_yawInitialized(false)
{
    loadParamInConfig();

    // 设置观测矩阵 H (4x8)
    cv::Mat H = cv::Mat::zeros(4, 8, CV_64F);
    H.at<double>(0,0) = 1.0;   // 观测 x
    H.at<double>(1,1) = 1.0;   // 观测 y
    H.at<double>(2,2) = 1.0;   // 观测 z
    H.at<double>(3,6) = 1.0;   // 观测 yaw
    m_ekf->setMeasurementMatrix(H);

    // 设置过程噪声 Q (8x8)
    const auto& c = Config::get().kalman;
    cv::Mat Q = cv::Mat::eye(8, 8, CV_64F) * 1e-4;
    Q.at<double>(0,0) = c.processNoisePos;
    Q.at<double>(1,1) = c.processNoisePos;
    Q.at<double>(2,2) = c.processNoisePos;
    Q.at<double>(3,3) = c.processNoiseVel;
    Q.at<double>(4,4) = c.processNoiseVel;
    Q.at<double>(5,5) = c.processNoiseVel;
    Q.at<double>(6,6) = c.yawProcessNoisePos;   // 新增 yaw 噪声参数
    Q.at<double>(7,7) = c.yawProcessNoiseVel;
    m_ekf->setProcessNoiseCov(Q);

    // 设置观测噪声 R (4x4)
    cv::Mat R = cv::Mat::eye(4, 4, CV_64F) * 1e-2;
    R.at<double>(0,0) = c.measurementNoisePos;
    R.at<double>(1,1) = c.measurementNoisePos;
    R.at<double>(2,2) = c.measurementNoisePos;
    R.at<double>(3,3) = c.yawMeasurementNoise;
    m_ekf->setMeasurementNoiseCov(R);

    // 初始误差协方差 P (8x8)
    cv::Mat P = cv::Mat::eye(8, 8, CV_64F) * c.initialErrorCov;
    m_ekf->setErrorCov(P);
}

void KalmanTracker::setTransitionMatrix(double dt) {
    cv::Mat F = cv::Mat::eye(8, 8, CV_64F);
    // 位置部分
    F.at<double>(0,3) = dt;
    F.at<double>(1,4) = dt;
    F.at<double>(2,5) = dt;
    // 角度部分
    F.at<double>(6,7) = dt;
    m_ekf->setTransitionMatrix(F);
}

void KalmanTracker::init(const cv::Point3f& position, double timeStamp) {
    cv::Mat x = cv::Mat::zeros(8, 1, CV_64F);
    x.at<double>(0) = position.x;
    x.at<double>(1) = position.y;
    x.at<double>(2) = position.z;
    // 速度初始为 0
    // yaw 和 yaw_vel 暂不初始化，等待第一次观测
    m_ekf->setState(x);

    m_lastTime = timeStamp;
    m_predictedPose = position;
    m_initialized = true;
}

cv::Point3f KalmanTracker::predict(double timeStamp) {
    if(!m_initialized) return cv::Point3f(0,0,0);

    if(m_lastTime > 0){
        m_dt = timeStamp - m_lastTime;
        if(m_dt > 0.1) m_dt = 0.033;
        if(m_dt < 0.001) m_dt = 0.033;
    } else {
        m_dt = 0.033;
    }

    setTransitionMatrix(m_dt);
    cv::Mat x = m_ekf->predict();

    m_predictedPose.x = x.at<double>(0);
    m_predictedPose.y = x.at<double>(1);
    m_predictedPose.z = x.at<double>(2);

    return m_predictedPose;
}

cv::Point3f KalmanTracker::update(const cv::Point3f measuredPos, double timeStamp) {
    if(!m_initialized){
        init(measuredPos, timeStamp);
        return measuredPos;
    }

    predict(timeStamp);

    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = measuredPos.x;
    z.at<double>(1) = measuredPos.y;
    z.at<double>(2) = measuredPos.z;
    // 如果当前有 yaw 观测，使用它；否则使用上一时刻的值
    // 这里我们假设 update 时没有 yaw 观测，实际应单独调用 updateYaw
    // 简单起见，我们可以将 yaw 观测也通过参数传入，但为了兼容原接口，保留独立 yaw 更新。
    // 此处不更新 yaw，保持原值。
    cv::Mat x = m_ekf->correct(z);

    cv::Point3f estPos(x.at<double>(0), x.at<double>(1), x.at<double>(2));
    m_lastTime = timeStamp;
    return estPos;
}

// yaw 相关方法实现
void KalmanTracker::initYaw(double yaw, double timeStamp) {
    // 从当前 EKF 状态中取出 yaw 和角速度（若无，则初始化为 0）
    cv::Mat x = m_ekf->getState();
    x.at<double>(6) = yaw;
    x.at<double>(7) = 0.0;
    m_ekf->setState(x);

    m_lastYawTime = timeStamp;
    m_predictedYaw = yaw;
    m_yawInitialized = true;
}

double KalmanTracker::predictYaw(double timeStamp) {
    if(!m_yawInitialized) return 0.0;
    // 先调用一次位置预测（会同时预测 yaw），然后从状态中提取 yaw
    // 注意：由于位置和 yaw 共用同一个 EKF，predict 会更新整个状态
    // 但我们需要保持位置预测和 yaw 预测的一致性，因此应在主循环中统一调用 predict
    // 这里简单实现：如果还没到预测时间，直接返回当前 yaw；否则调用一次 predict
    double dt = timeStamp - m_lastYawTime;
    if(dt > 0.001) {
        setTransitionMatrix(dt);
        cv::Mat x = m_ekf->predict();
        m_predictedYaw = x.at<double>(6);
        m_lastYawTime = timeStamp;
    }
    return m_predictedYaw;
}

double KalmanTracker::updateYaw(double measuredYaw, double timeStamp) {
    if(!m_yawInitialized) {
        initYaw(measuredYaw, timeStamp);
        return measuredYaw;
    }
    // 先预测到当前时间
    double dt = timeStamp - m_lastYawTime;
    if(dt > 0.001) {
        setTransitionMatrix(dt);
        m_ekf->predict();
        m_lastYawTime = timeStamp;
    }
    // 构造观测向量（仅更新 yaw，位置观测保持原值？这里需要小心：我们想更新 yaw 但位置保持不变）
    // 正确做法是同时更新所有观测，但为保持原接口，我们单独更新 yaw
    // 获取当前状态
    cv::Mat x = m_ekf->getState();
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = x.at<double>(0);   // 保持位置不变
    z.at<double>(1) = x.at<double>(1);
    z.at<double>(2) = x.at<double>(2);
    z.at<double>(3) = measuredYaw;
    // 修正
    cv::Mat x_new = m_ekf->correct(z);
    double estYaw = x_new.at<double>(6);
    return estYaw;
}

void KalmanTracker::loadParamInConfig() {
    // 参数已在构造函数中直接从 Config::get().kalman 读取
    // 此函数保留为空，兼容原有接口
}

cv::Point3f KalmanTracker::getEstimatedPosition() const {
    if(!m_initialized) return cv::Point3f(0,0,0);
    cv::Mat x = m_ekf->getState();
    return cv::Point3f(
        static_cast<float>(x.at<double>(0)),
        static_cast<float>(x.at<double>(1)),
        static_cast<float>(x.at<double>(2))
    );
}

double KalmanTracker::getEstimatedYaw() const {
    if(!m_yawInitialized) return 0.0;
    cv::Mat x = m_ekf->getState();
    return x.at<double>(6);
}

void KalmanTracker::loadYawParamsFromConfig() {
    // 参数已在构造函数中直接从 Config::get().kalman 读取
    // 此函数保留为空，兼容原有接口
}