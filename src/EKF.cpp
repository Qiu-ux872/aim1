#include "EKF.hpp"
#include <iostream>

ExtendedKalmanFilter::ExtendedKalmanFilter(int stateDim, int measDim, bool useLinear)
    : m_stateDim(stateDim), m_measDim(measDim), m_useLinear(useLinear)
{
    m_x = cv::Mat::zeros(stateDim, 1, CV_64F);
    m_P = cv::Mat::eye(stateDim, stateDim, CV_64F);
    m_F = cv::Mat::eye(stateDim, stateDim, CV_64F);
    m_H = cv::Mat::zeros(measDim, stateDim, CV_64F);
    m_Q = cv::Mat::eye(stateDim, stateDim, CV_64F) * 1e-4;
    m_R = cv::Mat::eye(measDim, measDim, CV_64F) * 1e-2;
    m_I = cv::Mat::eye(stateDim, stateDim, CV_64F);
}

void ExtendedKalmanFilter::setTransitionMatrix(const cv::Mat& F) { m_F = F.clone(); }
void ExtendedKalmanFilter::setMeasurementMatrix(const cv::Mat& H) { m_H = H.clone(); }
void ExtendedKalmanFilter::setProcessNoiseCov(const cv::Mat& Q) { m_Q = Q.clone(); }
void ExtendedKalmanFilter::setMeasurementNoiseCov(const cv::Mat& R) { m_R = R.clone(); }

void ExtendedKalmanFilter::setStateTransitionFunc(StateTransitionFunc f) { m_f = f; }
void ExtendedKalmanFilter::setMeasurementFunc(MeasurementFunc h) { m_h = h; }
void ExtendedKalmanFilter::setStateJacobianFunc(JacobianFunc Jf) { m_Jf = Jf; }
void ExtendedKalmanFilter::setMeasurementJacobianFunc(JacobianFunc Jh) { m_Jh = Jh; }

void ExtendedKalmanFilter::setState(const cv::Mat& x) { m_x = x.clone(); }
void ExtendedKalmanFilter::setErrorCov(const cv::Mat& P) { m_P = P.clone(); }

cv::Mat ExtendedKalmanFilter::predict(double dt) {
    cv::Mat F_current;

    if (m_useLinear) {
        // 线性卡尔曼: x_k = F * x_{k-1}
        F_current = m_F;

        // 更新转移矩阵中的 dt 依赖项 (如果需要)
        // 这里假设 m_F 已经根据 dt 设置好

        m_x = F_current * m_x;
        m_P = F_current * m_P * F_current.t() + m_Q;

    } else {
        // 扩展卡尔曼: x_k = f(x_{k-1}, dt)
        if (!m_f) {
            std::cerr << "[EKF] 状态转移函数未设置!" << std::endl;
            return m_x;
        }

        cv::Mat x_pred = m_f(m_x, dt);

        // 计算雅可比矩阵 J_f = df/dx
        cv::Mat Jf;
        if (m_Jf) {
            Jf = m_Jf(m_x, dt);
        } else {
            // 数值雅可比 (如果未提供)
            Jf = numericalJacobian(m_f, m_x, dt);
        }

        m_x = x_pred;
        m_P = Jf * m_P * Jf.t() + m_Q;
    }

    return m_x;
}

cv::Mat ExtendedKalmanFilter::correct(const cv::Mat& z) {
    cv::Mat H_current, z_pred;

    if (m_useLinear) {
        // 线性卡尔曼: z = H * x
        H_current = m_H;
        z_pred = H_current * m_x;

    } else {
        // 扩展卡尔曼: z = h(x)
        if (!m_h) {
            std::cerr << "[EKF] 观测函数未设置!" << std::endl;
            return m_x;
        }

        z_pred = m_h(m_x);

        // 计算雅可比矩阵 J_h = dh/dx
        if (m_Jh) {
            H_current = m_Jh(m_x, 0.0);
        } else {
            // 数值雅可比
            auto h_wrapper = [this](const cv::Mat& x, double dt) -> cv::Mat {
                return m_h(x);
            };
            H_current = numericalJacobian(h_wrapper, m_x, 0.0);
        }
    }

    // 卡尔曼增益
    cv::Mat S = H_current * m_P * H_current.t() + m_R;
    cv::Mat K = m_P * H_current.t() * S.inv();

    // 状态更新
    cv::Mat y = z - z_pred;  // 新息
    m_x = m_x + K * y;

    // 协方差更新
    m_P = (m_I - K * H_current) * m_P * (m_I - K * H_current).t() + K * m_R * K.t();

    return m_x;
}

// 数值雅可比计算 (用于扩展卡尔曼)
cv::Mat ExtendedKalmanFilter::numericalJacobian(
    std::function<cv::Mat(const cv::Mat&, double)> func,
    const cv::Mat& x,
    double dt)
{
    const double eps = 1e-6;
    cv::Mat J = cv::Mat::zeros(m_measDim, m_stateDim, CV_64F);

    cv::Mat x_perturbed = x.clone();
    cv::Mat f_base = func(x, dt);

    for (int j = 0; j < m_stateDim; ++j) {
        // 正向扰动
        x_perturbed.at<double>(j) = x.at<double>(j) + eps;
        cv::Mat f_plus = func(x_perturbed, dt);

        // 计算数值导数: (f(x+eps) - f(x)) / eps
        for (int i = 0; i < (m_useLinear ? m_stateDim : m_measDim); ++i) {
            J.at<double>(i, j) = (f_plus.at<double>(i) - f_base.at<double>(i)) / eps;
        }

        // 恢复原始值
        x_perturbed.at<double>(j) = x.at<double>(j);
    }

    return J;
}
