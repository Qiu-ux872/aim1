#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "CameraApi.h"
#include "Config.hpp"

using namespace std;
using namespace cv;

class CameraDriver{
public:
    CameraDriver();
    ~CameraDriver();

    // 打开相机
    bool open();

    // 关闭相机
    void close();

    // 开始采集图像
    bool start();

    // 停止采集图像
    void stop();

    // 捕获一帧图像（返回 RGB格式）
    Mat capture(int timeoutMs = 1000);

    // 获取原始图像缓冲区
    unsigned char* captureRaw(int& width, int& height, int timeoutMs = 1000);

    // 设置曝光时间（ms）
    bool setExposureTime(float exposure);

    // 设置模拟增益
    bool setAnalogGain(float gain);

    // 设置自动曝光
    bool setAutoExposure(bool enable);

    // 获取相机句柄
    CameraHandle handle() const { return m_hCamera; }

    // 获取相机特性信息
    const tSdkCameraCapbility& cability() const { return m_capability; }

private:
    CameraHandle          m_hCamera;     // 相机句柄
    tSdkCameraDevInfo    m_devInfo;     // 相机设备信息
    tSdkCameraCapbility  m_capability;  // 相机特性信息
    bool                  m_isOpen;      // 是否已打开相机
    bool                  m_isStreaming; // 是否正在采集图像
    int                   m_width;       // 图像宽度
    int                   m_height;      // 图像高度
    // 打印错误信息
    void printError(CameraSdkStatus status, const char* msg);

};