#include "KalmanTracker.hpp"
#include <cmath>
#include <iostream>

KalmanTracker::KalmanTracker()
    : m_kf(std::make_unique<EKF>(8, 4, false)),
      m_lastTimePos(0),
      m_lastTimeYaw(0),
      m_dt(0),
      m_predictedPose(0, 0, 0),
      m_predictedYaw(0),
      m_initialized(false),
      m_yawInitialized(false)
{
    loadParamsFromConfig();

    // 观测矩阵 H (仅用于线性模式，EKF 下使用雅可比)
    cv::Mat H = cv::Mat::zeros(4, 8, CV_64F);
    H.at<double>(0,0) = 1.0;
    H.at<double>(1,1) = 1.0;
    H.at<double>(2,2) = 1.0;
    H.at<double>(3,6) = 1.0;
    m_kf->setMeasurementMatrix(H);

    // 过程噪声 Q
    const auto& c = Config::get().kalman;
    cv::Mat Q = cv::Mat::eye(8, 8, CV_64F) * 1e-4;
    Q.at<double>(0,0) = c.processNoisePos;
    Q.at<double>(1,1) = c.processNoisePos;
    Q.at<double>(2,2) = c.processNoisePos;
    Q.at<double>(3,3) = c.processNoiseVel;
    Q.at<double>(4,4) = c.processNoiseVel;
    Q.at<double>(5,5) = c.processNoiseVel;
    Q.at<double>(6,6) = c.yawProcessNoisePos;
    Q.at<double>(7,7) = c.yawProcessNoiseVel;
    m_kf->setProcessNoiseCov(Q);

    // 观测噪声 R
    cv::Mat R = cv::Mat::eye(4, 4, CV_64F) * 1e-2;
    R.at<double>(0,0) = c.measurementNoisePos;
    R.at<double>(1,1) = c.measurementNoisePos;
    R.at<double>(2,2) = c.measurementNoisePos;
    R.at<double>(3,3) = c.yawMeasurementNoise;
    m_kf->setMeasurementNoiseCov(R);

    // 初始协方差 P
    cv::Mat P = cv::Mat::eye(8, 8, CV_64F) * c.initialErrorCov;
    m_kf->setErrorCov(P);

    // ========== 设置非线性函数 ==========
    // 状态转移函数
    m_kf->setStateTransitionFunc([this](const cv::Mat& x, double dt) -> cv::Mat {
        cv::Mat x_new = x.clone();
        x_new.at<double>(0) = x.at<double>(0) + x.at<double>(3) * dt;
        x_new.at<double>(1) = x.at<double>(1) + x.at<double>(4) * dt;
        x_new.at<double>(2) = x.at<double>(2) + x.at<double>(5) * dt;
        x_new.at<double>(6) = x.at<double>(6) + x.at<double>(7) * dt;
        return x_new;
    });

    // 状态雅可比
    m_kf->setStateJacobianFunc([this](const cv::Mat& x, double dt) -> cv::Mat {
        cv::Mat J = cv::Mat::eye(8, 8, CV_64F);
        J.at<double>(0,3) = dt;
        J.at<double>(1,4) = dt;
        J.at<double>(2,5) = dt;
        J.at<double>(6,7) = dt;
        return J;
    });

    // 观测函数
    m_kf->setMeasurementFunc([this](const cv::Mat& x) -> cv::Mat {
        cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
        z.at<double>(0) = x.at<double>(0);
        z.at<double>(1) = x.at<double>(1);
        z.at<double>(2) = x.at<double>(2);
        z.at<double>(3) = x.at<double>(6);
        return z;
    });

    // 观测雅可比
    m_kf->setMeasurementJacobianFunc([this](const cv::Mat& x, double dt) -> cv::Mat {
        cv::Mat H = cv::Mat::zeros(4, 8, CV_64F);
        H.at<double>(0,0) = 1.0;
        H.at<double>(1,1) = 1.0;
        H.at<double>(2,2) = 1.0;
        H.at<double>(3,6) = 1.0;
        return H;
    });

    std::cout << "[Kalman] 初始化完成，参数: "
              << "posNoise=" << c.processNoisePos
              << ", velNoise=" << c.processNoiseVel
              << ", measNoise=" << c.measurementNoisePos
              << ", yawPosNoise=" << c.yawProcessNoisePos
              << ", yawVelNoise=" << c.yawProcessNoiseVel
              << ", yawMeasNoise=" << c.yawMeasurementNoise << std::endl;
}

void KalmanTracker::loadParamsFromConfig() {}

void KalmanTracker::init(const cv::Point3f& position, double timeStamp) {
    cv::Mat x = cv::Mat::zeros(8, 1, CV_64F);
    x.at<double>(0) = position.x;
    x.at<double>(1) = position.y;
    x.at<double>(2) = position.z;
    m_kf->setState(x);
    m_lastTimePos = timeStamp;
    m_predictedPose = position;
    m_initialized = true;
}

cv::Point3f KalmanTracker::predict(double timeStamp) {
    if(!m_initialized) return cv::Point3f(0,0,0);
    if(m_lastTimePos > 0){
        m_dt = timeStamp - m_lastTimePos;
        if(m_dt > 0.1) m_dt = 0.033;
        if(m_dt < 0.001) m_dt = 0.033;
    } else {
        m_dt = 0.033;
    }
    cv::Mat x = m_kf->predict(m_dt);
    m_predictedPose.x = x.at<double>(0);
    m_predictedPose.y = x.at<double>(1);
    m_predictedPose.z = x.at<double>(2);
    m_lastTimePos = timeStamp;
    return m_predictedPose;
}

cv::Point3f KalmanTracker::update(const cv::Point3f measuredPos, double timeStamp) {
    if(!m_initialized){
        init(measuredPos, timeStamp);
        return measuredPos;
    }
    predict(timeStamp);
    cv::Mat x_current = m_kf->getState();
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = measuredPos.x;
    z.at<double>(1) = measuredPos.y;
    z.at<double>(2) = measuredPos.z;
    z.at<double>(3) = x_current.at<double>(6);
    cv::Mat x = m_kf->correct(z);
    cv::Point3f estPos(x.at<double>(0), x.at<double>(1), x.at<double>(2));
    return estPos;
}

void KalmanTracker::initYaw(double yaw, double timeStamp) {
    cv::Mat x = m_kf->getState();
    x.at<double>(6) = yaw;
    x.at<double>(7) = 0.0;
    m_kf->setState(x);
    m_lastTimeYaw = timeStamp;
    m_predictedYaw = yaw;
    m_yawInitialized = true;
}

double KalmanTracker::predictYaw(double timeStamp) {
    if(!m_yawInitialized) return 0.0;
    double dt = timeStamp - m_lastTimeYaw;
    if(dt < 0.001) dt = 0.033;
    if(dt > 0.1) dt = 0.033;
    cv::Mat x = m_kf->predict(dt);
    m_predictedYaw = x.at<double>(6);
    m_lastTimeYaw = timeStamp;
    return m_predictedYaw;
}

double KalmanTracker::updateYaw(double measuredYaw, double timeStamp) {
    if(!m_yawInitialized) {
        initYaw(measuredYaw, timeStamp);
        return measuredYaw;
    }
    double dt = timeStamp - m_lastTimeYaw;
    if(dt < 0.001) dt = 0.033;
    if(dt > 0.1) dt = 0.033;
    m_kf->predict(dt);
    cv::Mat x_current = m_kf->getState();
    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F);
    z.at<double>(0) = x_current.at<double>(0);
    z.at<double>(1) = x_current.at<double>(1);
    z.at<double>(2) = x_current.at<double>(2);
    z.at<double>(3) = measuredYaw;
    cv::Mat x_new = m_kf->correct(z);
    double estYaw = x_new.at<double>(6);
    m_lastTimeYaw = timeStamp;
    return estYaw;
}

cv::Point3f KalmanTracker::getEstimatedPosition() const {
    if(!m_initialized) return cv::Point3f(0,0,0);
    cv::Mat x = m_kf->getState();
    return cv::Point3f(x.at<double>(0), x.at<double>(1), x.at<double>(2));
}

double KalmanTracker::getEstimatedYaw() const {
    if(!m_yawInitialized) return 0.0;
    cv::Mat x = m_kf->getState();
    return x.at<double>(6);
}