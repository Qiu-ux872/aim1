#pragma once

#include <opencv2/opencv.hpp>

class EKF {
public:
    EKF(int stateDim, int measDim);
    void setTransitionMatrix(const cv::Mat& F);
    void setMeasurementMatrix(const cv::Mat& H);
    void setProcessNoiseCov(const cv::Mat& Q);
    void setMeasurementNoiseCov(const cv::Mat& R);
    void setState(const cv::Mat& x);
    void setErrorCov(const cv::Mat& P);

    cv::Mat predict();
    cv::Mat correct(const cv::Mat& z);

    cv::Mat getState() const { return m_x; }
    cv::Mat getErrorCov() const { return m_P; }

private:
    int m_stateDim;
    int m_measDim;
    cv::Mat m_x;       // 状态向量
    cv::Mat m_P;       // 误差协方差
    cv::Mat m_F;       // 状态转移矩阵
    cv::Mat m_H;       // 观测矩阵
    cv::Mat m_Q;       // 过程噪声协方差
    cv::Mat m_R;       // 观测噪声协方差
    cv::Mat m_I;       // 单位矩阵
};