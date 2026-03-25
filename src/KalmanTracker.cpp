#include "Config.hpp"
#include "KalmanTracker.hpp"
#include <cmath>

using namespace std;
using namespace cv;

KalmanTracker::KalmanTracker()
    : m_kf(6, 3, 0),
      m_lastTime(0),
      m_dt(0),
      m_initialized(false),
      m_kfYaw(2, 1, 0),          // 2维状态，1维观测
      m_lastYawTime(0),
      m_dtYaw(0),
      m_predictedYaw(0),
      m_yawInitialized(false)
{
    // 位置滤波器观测矩阵 H(3x6)
    m_kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
    m_kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    m_kf.measurementMatrix.at<float>(1, 1) = 1.0f;
    m_kf.measurementMatrix.at<float>(2, 2) = 1.0f;

    loadParamInConfig();

    setIdentity(m_kf.errorCovPost, Scalar(Config::get().kalman.initialErrorCov));

    loadYawParamsFromConfig();

    // 观测矩阵 H(1x2)
    m_kfYaw.measurementMatrix = cv::Mat::zeros(1, 2, CV_32F);
    m_kfYaw.measurementMatrix.at<float>(0, 0) = 1.0f;   // 直接观测 yaw

    // 后验误差协方差 P 初始值
    setIdentity(m_kfYaw.errorCovPost, Scalar(1.0));
}

void KalmanTracker::loadParamInConfig(){
    const auto& c = Config::get().kalman;
    setIdentity(m_kf.processNoiseCov, Scalar(0));
    m_kf.processNoiseCov.at<float>(0,0) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(1,1) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(2,2) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(3,3) = c.processNoiseVel;
    m_kf.processNoiseCov.at<float>(4,4) = c.processNoiseVel;
    m_kf.processNoiseCov.at<float>(5,5) = c.processNoiseVel;

    setIdentity(m_kf.measurementNoiseCov, Scalar(c.measurementNoisePos));
}

void KalmanTracker::setTransitionMatrix(double dt){
    Mat& F = m_kf.transitionMatrix;
    setIdentity(F);
    F.at<float>(0,3) = dt;
    F.at<float>(1,4) = dt;
    F.at<float>(2,5) = dt;
}

void KalmanTracker::init(const Point3f& position, double timeStamp){
    m_state = cv::Mat::zeros(6, 1, CV_32F);
    m_state.at<float>(0) = position.x;
    m_state.at<float>(1) = position.y;
    m_state.at<float>(2) = position.z;
    m_kf.statePost = m_state.clone();

    m_lastTime = timeStamp;
    m_predictedPose = position;
    m_initialized = true;
}

Point3f KalmanTracker::predict(double timeStamp){
    if(!m_initialized) return Point3f(0,0,0);

    if(m_lastTime > 0){
        m_dt = timeStamp - m_lastTime;
        if(m_dt > 0.1) m_dt = 0.033;
        if(m_dt < 0.001) m_dt = 0.033;
    } else {
        m_dt = 0.033;
    }

    setTransitionMatrix(m_dt);
    m_state = m_kf.predict();

    m_predictedPose.x = m_state.at<float>(0);
    m_predictedPose.y = m_state.at<float>(1);
    m_predictedPose.z = m_state.at<float>(2);

    return m_predictedPose;
}

Point3f KalmanTracker::update(const Point3f measuredPos, double timeStamp){
    if(!m_initialized){
        init(measuredPos, timeStamp);
        return measuredPos;
    }

    predict(timeStamp);

    m_measured = cv::Mat::zeros(3, 1, CV_32F);
    m_measured.at<float>(0) = measuredPos.x;
    m_measured.at<float>(1) = measuredPos.y;
    m_measured.at<float>(2) = measuredPos.z;

    m_state = m_kf.correct(m_measured);

    Point3f estPos(m_state.at<float>(0), m_state.at<float>(1), m_state.at<float>(2));
    m_lastTime = timeStamp;
    return estPos;
}

Point3f KalmanTracker::getEstimatedPosition() const {
    if(!m_initialized) return cv::Point3f(0,0,0);
    return cv::Point3f(m_kf.statePost.at<float>(0),
                       m_kf.statePost.at<float>(1),
                       m_kf.statePost.at<float>(2));
}

// ==================== 新增 yaw 滤波器实现 ====================
void KalmanTracker::loadYawParamsFromConfig() {
    const auto& c = Config::get().kalman;

    // 过程噪声协方差 Q(2x2) 对角阵
    setIdentity(m_kfYaw.processNoiseCov, Scalar(0));
    // 位置过程噪声（角度）和速度过程噪声（角速度）
    m_kfYaw.processNoiseCov.at<float>(0,0) = c.yawProcessNoisePos;   // 需要在 KalmanConfig 中添加
    m_kfYaw.processNoiseCov.at<float>(1,1) = c.yawProcessNoiseVel;

    // 观测噪声 R(1x1)
    setIdentity(m_kfYaw.measurementNoiseCov, Scalar(c.yawMeasurementNoise));
}

void KalmanTracker::setYawTransitionMatrix(double dt) {
    Mat& F = m_kfYaw.transitionMatrix;
    setIdentity(F);
    F.at<float>(0,1) = dt;   // 角度 += 角速度 * dt
    // 角速度项自身系数为1
}

void KalmanTracker::initYaw(double yaw, double timeStamp) {
    Mat state = cv::Mat::zeros(2, 1, CV_32F);
    state.at<float>(0) = yaw;
    state.at<float>(1) = 0.0f;   // 初始角速度为0
    m_kfYaw.statePost = state.clone();

    m_lastYawTime = timeStamp;
    m_predictedYaw = yaw;
    m_yawInitialized = true;
}

double KalmanTracker::predictYaw(double timeStamp) {
    if(!m_yawInitialized) return 0.0;

    if(m_lastYawTime > 0){
        m_dtYaw = timeStamp - m_lastYawTime;
        if(m_dtYaw > 0.1) m_dtYaw = 0.033;
        if(m_dtYaw < 0.001) m_dtYaw = 0.033;
    } else {
        m_dtYaw = 0.033;
    }

    setYawTransitionMatrix(m_dtYaw);
    Mat state = m_kfYaw.predict();

    m_predictedYaw = state.at<float>(0);
    return m_predictedYaw;
}

double KalmanTracker::updateYaw(double measuredYaw, double timeStamp) {
    if(!m_yawInitialized) {
        initYaw(measuredYaw, timeStamp);
        return measuredYaw;
    }

    // 先预测（得到当前时刻的预测）
    predictYaw(timeStamp);

    Mat measurement = cv::Mat::zeros(1, 1, CV_32F);
    measurement.at<float>(0) = measuredYaw;

    Mat state = m_kfYaw.correct(measurement);

    double estYaw = state.at<float>(0);
    m_lastYawTime = timeStamp;
    return estYaw;
}

double KalmanTracker::getEstimatedYaw() const {
    if(!m_yawInitialized) return 0.0;
    return m_kfYaw.statePost.at<float>(0);
}