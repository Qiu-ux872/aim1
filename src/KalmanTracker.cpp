#include "Config.hpp"
#include "KalmanTracker.hpp"
#include <cmath>

using namespace std;
using namespace cv;

KalmanTracker::KalmanTracker()
    : m_kf(6, 3, 0),            // 6维状态 3维观测 无控制
    m_lastTime(0),
    m_dt(0),
    m_initialized(false)
{
    // 观测矩阵 H(3x6)
    m_kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
    m_kf.measurementMatrix.at<float>(0, 0) = 1.0f; // x
    m_kf.measurementMatrix.at<float>(1, 1) = 1.0f; // y
    m_kf.measurementMatrix.at<float>(2, 2) = 1.0f; // z

    // 从配置中加载噪声参数
    loadParamInConfig();

    // 后验证误差协方差矩阵 P(6x6)
    setIdentity(m_kf.errorCovPost, Scalar(Config::get().kalman.initialErrorCov));
}

void KalmanTracker::loadParamInConfig(){
    const auto& c = Config::get().kalman;

    // 过程噪声协方差矩阵 Q(6x6)
    setIdentity(m_kf.processNoiseCov, Scalar(0));   // 清零
    m_kf.processNoiseCov.at<float>(0, 0) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(1, 1) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(2, 2) = c.processNoisePos;
    m_kf.processNoiseCov.at<float>(3, 3) = c.processNoiseVel;
    m_kf.processNoiseCov.at<float>(4, 4) = c.processNoiseVel;
    m_kf.processNoiseCov.at<float>(5, 5) = c.processNoiseVel;

    // 预测噪声协方差 R(3x3)
    setIdentity(m_kf.measurementNoiseCov, Scalar(c.measurementNoisePos));
}

void KalmanTracker::setTransitionMatrix(double dt){
    // 状态转移矩阵 F(6x6)
    Mat& F = m_kf.transitionMatrix;
    setIdentity(F);
    F.at<float>(0, 3) = dt;
    F.at<float>(1, 4) = dt;
    F.at<float>(2, 5) = dt;
}

void KalmanTracker::init(const Point3f& position, double timeStamp){
    // 初始化状态向量 [x, y, z, vx, vy, vz]
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
    if(!m_initialized){
        return Point3f(0, 0, 0);
    }

    // 计算时间间隔
    if(m_lastTime > 0){
        m_dt = timeStamp - m_lastTime;
        // 限制最大时间间隔 防止dt过大导致发散
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

    //先预测
    predict(timeStamp);

    m_measured = cv::Mat::zeros(3, 1, CV_32F);
    m_measured.at<float>(0) = measuredPos.x;
    m_measured.at<float>(1) = measuredPos.y;
    m_measured.at<float>(2) = measuredPos.z;

    m_state = m_kf.correct(m_measured);

    Point3f estPos(
        m_state.at<float>(0),
        m_state.at<float>(1),
        m_state.at<float>(2)
    );

    m_lastTime = timeStamp;
    return estPos;
}

Point3f KalmanTracker::getEstimatedPosition() const {
    if (!m_initialized) return cv::Point3f(0,0,0);
    return cv::Point3f(
        m_kf.statePost.at<float>(0),
        m_kf.statePost.at<float>(1),
        m_kf.statePost.at<float>(2)
    );
}