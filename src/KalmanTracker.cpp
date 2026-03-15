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