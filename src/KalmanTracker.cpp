#include "KalmanTracker.hpp"
#include <cmath>

KalmanTracker::KalmanTracker()
    : m_kf(std::make_unique<EKF>(8, 4, true)),    // 8维状态，4维观测，使用线性KF
      m_lastTime(0),
      m_dt(0),
      m_predictedPose(0, 0, 0),
      m_predictedYaw(0),
      m_initialized(false),
      m_yawInitialized(false)
{
    loadParamsFromConfig();

    // 设置观测矩阵 H (4x8)
    cv::Mat H = cv::Mat::zeros(4, 8, CV_64F);
    H.at<double>(0,0) = 1.0;   // 观测 x
    H.at<double>(1,1) = 1.0;   // 观测 y
    H.at<double>(2,2) = 1.0;   // 观测 z
    H.at<double>(3,6) = 1.0;   // 观测 yaw
    m_kf->setMeasurementMatrix(H);

    // 设置过程噪声 Q (8x8)
    const auto& c = Config::get().kalman;
    cv::Mat Q = cv::Mat::eye(8, 8, CV_64F) * 1e-4;
    Q.at<double>(0,0) = c.processNoisePos;
    Q.at<double>(1,1) = c.processNoisePos;
    Q.at<double>(2,2) = c.processNoisePos;
    Q.at<double>(3,3) = c.processNoiseVel;
    Q.at<double>(4,4) = c.processNoiseVel;
    Q.at<double>(5,5) = c.processNoiseVel;
    Q.at<double>(6,6) = c.yawProcessNoisePos;   // yaw 噪声参数
    Q.at<double>(7,7) = c.yawProcessNoiseVel;
    m_kf->setProcessNoiseCov(Q);

    // 设置观测噪声 R (4x4)
    cv::Mat R = cv::Mat::eye(4, 4, CV_64F) * 1e-2;
    R.at<double>(0,0) = c.measurementNoisePos;
    R.at<double>(1,1) = c.measurementNoisePos;
    R.at<double>(2,2) = c.measurementNoisePos;
    R.at<double>(3,3) = c.yawMeasurementNoise;
    m_kf->setMeasurementNoiseCov(R);

    // 初始误差协方差 P (8x8)
    cv::Mat P = cv::Mat::eye(8, 8, CV_64F) * c.initialErrorCov;
    m_kf->setErrorCov(P);
}

void KalmanTracker::setTransitionMatrix(double dt) {
    cv::Mat F = cv::Mat::eye(8, 8, CV_64F);
    // 位置部分: x = x + vx*dt
    F.at<double>(0,3) = dt;
    F.at<double>(1,4) = dt;
    F.at<double>(2,5) = dt;
    // Yaw 部分: yaw = yaw + yaw_vel*dt
    F.at<double>(6,7) = dt;
    m_kf->setTransitionMatrix(F);
}

void KalmanTracker::init(const cv::Point3f& position, double timeStamp) {
    cv::Mat x = cv::Mat::zeros(8, 1, CV_64F);
    x.at<double>(0) = position.x;
    x.at<double>(1) = position.y;
    x.at<double>(2) = position.z;
    // 速度和 yaw 初始为 0
    m_kf->setState(x);

    m_lastTime = timeStamp;
    m_predictedPose = position;
    m_initialized = true;
}

cv::Point3f KalmanTracker::predict(double timeStamp) {
    if(!m_initialized) return cv::Point3f(0,0,0);

    // 计算 dt
    if(m_lastTime > 0){
        m_dt = timeStamp - m_lastTime;
        if(m_dt > 0.1) m_dt = 0.033;  // 限制最大 dt
        if(m_dt < 0.001) m_dt = 0.033;  // 限制最小 dt
    } else {
        m_dt = 0.033;
    }

    setTransitionMatrix(m_dt);
    cv::Mat x = m_kf->predict(m_dt);

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

    // 构造观测向量 (仅更新位置,yaw保持原值)
    cv::Mat x_current = m_kf->getState();
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = measuredPos.x;
    z.at<double>(1) = measuredPos.y;
    z.at<double>(2) = measuredPos.z;
    z.at<double>(3) = x_current.at<double>(6);  // 保持 yaw 不变

    cv::Mat x = m_kf->correct(z);

    cv::Point3f estPos(x.at<double>(0), x.at<double>(1), x.at<double>(2));
    m_lastTime = timeStamp;
    return estPos;
}

void KalmanTracker::initYaw(double yaw, double timeStamp) {
    // 更新当前状态中的 yaw
    cv::Mat x = m_kf->getState();
    x.at<double>(6) = yaw;
    x.at<double>(7) = 0.0;  // 角速度初始为 0
    m_kf->setState(x);

    m_lastTime = timeStamp;
    m_predictedYaw = yaw;
    m_yawInitialized = true;
}

double KalmanTracker::predictYaw(double timeStamp) {
    if(!m_yawInitialized) return 0.0;

    double dt = timeStamp - m_lastTime;
    if(dt > 0.001) {
        setTransitionMatrix(dt);
        cv::Mat x = m_kf->predict(dt);
        m_predictedYaw = x.at<double>(6);
        m_lastTime = timeStamp;
    }
    return m_predictedYaw;
}

double KalmanTracker::updateYaw(double measuredYaw, double timeStamp) {
    if(!m_yawInitialized) {
        initYaw(measuredYaw, timeStamp);
        return measuredYaw;
    }

    // 先预测到当前时间
    double dt = timeStamp - m_lastTime;
    if(dt > 0.001) {
        setTransitionMatrix(dt);
        m_kf->predict(dt);
        m_lastTime = timeStamp;
    }

    // 构造观测向量 (仅更新 yaw,位置保持原值)
    cv::Mat x_current = m_kf->getState();
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = x_current.at<double>(0);
    z.at<double>(1) = x_current.at<double>(1);
    z.at<double>(2) = x_current.at<double>(2);
    z.at<double>(3) = measuredYaw;

    cv::Mat x_new = m_kf->correct(z);
    double estYaw = x_new.at<double>(6);
    return estYaw;
}

cv::Point3f KalmanTracker::getEstimatedPosition() const {
    if(!m_initialized) return cv::Point3f(0,0,0);
    cv::Mat x = m_kf->getState();
    return cv::Point3f(
        static_cast<float>(x.at<double>(0)),
        static_cast<float>(x.at<double>(1)),
        static_cast<float>(x.at<double>(2))
    );
}

double KalmanTracker::getEstimatedYaw() const {
    if(!m_yawInitialized) return 0.0;
    cv::Mat x = m_kf->getState();
    return x.at<double>(6);
}

void KalmanTracker::loadParamsFromConfig() {
    // 参数已在构造函数中直接从 Config::get().kalman 读取
    // 此函数保留为空，兼容原有接口
}