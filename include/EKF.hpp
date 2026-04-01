#pragma once

#include <opencv2/opencv.hpp>
#include <functional>

class ExtendedKalmanFilter {
public:
    // 构造函数
    ExtendedKalmanFilter(int stateDim, int measDim, bool useLinear = true);

    // ===== 线性卡尔曼滤波器接口 =====
    void setTransitionMatrix(const cv::Mat& F);
    void setMeasurementMatrix(const cv::Mat& H);
    void setProcessNoiseCov(const cv::Mat& Q);
    void setMeasurementNoiseCov(const cv::Mat& R);

    // ===== 扩展卡尔曼滤波器接口 (非线性) =====
    // 设置非线性状态转移函数: x_k = f(x_{k-1}, dt)
    using StateTransitionFunc = std::function<cv::Mat(const cv::Mat& x, double dt)>;
    void setStateTransitionFunc(StateTransitionFunc f);

    // 设置非线性观测函数: z = h(x)
    using MeasurementFunc = std::function<cv::Mat(const cv::Mat& x)>;
    void setMeasurementFunc(MeasurementFunc h);

    // 设置雅可比矩阵计算函数: J_f = df/dx, J_h = dh/dx
    using JacobianFunc = std::function<cv::Mat(const cv::Mat& x, double dt)>;
    void setStateJacobianFunc(JacobianFunc Jf);
    void setMeasurementJacobianFunc(JacobianFunc Jh);

    // ===== 通用接口 =====
    void setState(const cv::Mat& x);
    void setErrorCov(const cv::Mat& P);

    cv::Mat predict(double dt = 1.0);
    cv::Mat correct(const cv::Mat& z);

    cv::Mat getState() const { return m_x; }
    cv::Mat getErrorCov() const { return m_P; }
    bool isLinear() const { return m_useLinear; }

private:
    int m_stateDim;
    int m_measDim;
    bool m_useLinear;  // true=线性KF, false=扩展EKF

    cv::Mat m_x;       // 状态向量
    cv::Mat m_P;       // 误差协方差
    cv::Mat m_F;       // 状态转移矩阵 (线性)
    cv::Mat m_H;       // 观测矩阵 (线性)
    cv::Mat m_Q;       // 过程噪声协方差
    cv::Mat m_R;       // 观测噪声协方差
    cv::Mat m_I;       // 单位矩阵

    // 非线性函数
    StateTransitionFunc m_f;    // 状态转移函数
    MeasurementFunc m_h;       // 观测函数
    JacobianFunc m_Jf;         // 状态雅可比矩阵函数
    JacobianFunc m_Jh;         // 观测雅可比矩阵函数

    // 数值雅可比计算
    cv::Mat numericalJacobian(
        std::function<cv::Mat(const cv::Mat&, double)> func,
        const cv::Mat& x,
        double dt);
};

// 保留 EKF 作为别名以兼容旧代码
using EKF = ExtendedKalmanFilter;