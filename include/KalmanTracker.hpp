#pragma once

#include <opencv2/opencv.hpp>
#include "Param.hpp"
#include "Config.hpp"
        
using namespace std;
using namespace cv;

class KalmanTracker{
public:
    KalmanTracker();

    // 初始化状态（第一次观测到目标时调用）
    void init(const Point3f& position, double timeStamp);

    // 预测下一时间位置
    Point3f predict(double timeStamp);

    // 更新滤波器 (传入当前观测)
    Point3f update(const Point3f measuredPos, double timeStamp);

    // 获取当前最优估计的位置
    Point3f getEstimatedPosition() const;

    // 获取预测的下一时刻位置（用于绘制）
    Point3f getPredictionPosition() const { return m_predictedPose; }

    // 检查是否已初始化
    bool isInitialized() const { return m_initialized; }    

private:
    KalmanFilter m_kf;          // 卡尔曼滤波器对象 (6维状态，3维观测)
    Mat m_state;                // 状态向量（6x1）
    Mat m_measured;             // 观测向量（3x1）
    double m_lastTime;          // 上一帧时间戳（s）
    double m_dt;                // 时间间隔（s）
    Point3f m_predictedPose;    // 预测位置（用于绘制）
    bool m_initialized;         // 是否已初始化

    void setTransitionMatrix(double dt);
    void loadParamInConfig();

};