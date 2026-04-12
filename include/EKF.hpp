#pragma once

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter();
    ~ExtendedKalmanFilter() = default;

    // 初始化状态（位置和 yaw）
    void init(const cv::Point3f& position, double yaw, double timeStamp);

    // 预测下一时刻状态（位置和 yaw）
    void predict(double timeStamp);

    // 更新位置观测
    void updatePosition(const cv::Point3f& measuredPos, double timeStamp);

    // 更新 yaw 观测
    void updateYaw(double measuredYaw, double timeStamp);

    // 获取估计值
    cv::Point3f getEstimatedPosition() const;
    double getEstimatedYaw() const;

    // 获取预测值
    cv::Point3f getPredictedPosition() const { return m_predictedPose; }
    double getPredictedYaw() const { return m_predictedYaw; }

    bool isInitialized() const { return m_initialized; }
    bool isYawInitialized() const { return m_yawInitialized; }

private:
    // 卡尔曼滤波核心
    cv::Mat m_x;       // 状态向量 (8x1)
    cv::Mat m_P;       // 误差协方差 (8x8)
    cv::Mat m_F;       // 状态转移矩阵 (8x8)
    cv::Mat m_H;       // 观测矩阵 (4x8)
    cv::Mat m_Q;       // 过程噪声 (8x8)
    cv::Mat m_R;       // 观测噪声 (4x4)
    cv::Mat m_I;       // 单位矩阵 (8x8)

    // 非线性函数（线性化模型，实际为线性）
    cv::Mat stateTransition(const cv::Mat& x, double dt);
    cv::Mat measurementModel(const cv::Mat& x);
    cv::Mat stateJacobian(double dt);
    cv::Mat measurementJacobian();

    // 时间管理
    double m_lastTimePos;
    double m_lastTimeYaw;
    double m_dt;

    // 预测/估计结果缓存
    cv::Point3f m_predictedPose;
    double m_predictedYaw;

    bool m_initialized;
    bool m_yawInitialized;

    void loadParamsFromConfig();
    void setTransitionMatrix(double dt);
    void predictState(double dt);
    void correct(const cv::Mat& z);
};