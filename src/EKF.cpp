#include "EKF.hpp"
#include <iostream>

EKF::EKF(int stateDim, int measDim)
    : m_stateDim(stateDim), m_measDim(measDim)
{
    m_x = cv::Mat::zeros(stateDim, 1, CV_64F);
    m_P = cv::Mat::eye(stateDim, stateDim, CV_64F);
    m_F = cv::Mat::eye(stateDim, stateDim, CV_64F);
    m_H = cv::Mat::zeros(measDim, stateDim, CV_64F);
    m_Q = cv::Mat::eye(stateDim, stateDim, CV_64F) * 1e-4;
    m_R = cv::Mat::eye(measDim, measDim, CV_64F) * 1e-2;
    m_I = cv::Mat::eye(stateDim, stateDim, CV_64F);
}

void EKF::setTransitionMatrix(const cv::Mat& F) { m_F = F.clone(); }
void EKF::setMeasurementMatrix(const cv::Mat& H) { m_H = H.clone(); }
void EKF::setProcessNoiseCov(const cv::Mat& Q) { m_Q = Q.clone(); }
void EKF::setMeasurementNoiseCov(const cv::Mat& R) { m_R = R.clone(); }
void EKF::setState(const cv::Mat& x) { m_x = x.clone(); }
void EKF::setErrorCov(const cv::Mat& P) { m_P = P.clone(); }

cv::Mat EKF::predict() {
    // 状态预测
    m_x = m_F * m_x;
    // 协方差预测
    m_P = m_F * m_P * m_F.t() + m_Q;
    return m_x;
}

cv::Mat EKF::correct(const cv::Mat& z) {
    // 卡尔曼增益
    cv::Mat S = m_H * m_P * m_H.t() + m_R;
    cv::Mat K = m_P * m_H.t() * S.inv();

    // 状态更新
    cv::Mat y = z - m_H * m_x;
    m_x = m_x + K * y;

    // 协方差更新
    m_P = (m_I - K * m_H) * m_P;

    return m_x;
}
