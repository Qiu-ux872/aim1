#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

struct CameraConfig{
    int width = 640;
    int height = 480;
    int fps = 30;
    float exposure = 1000.0;
    float gain = 10.0;
    float rgb_gain_r = 1.0f;    // 红色通道增益
    float rgb_gain_g = 1.0f;    // 绿色通道增益
    float rgb_gain_b = 1.0f;    // 蓝色通道增益
};

struct SerialConfig{
    string port = "/dev/ttyUSB0";
    int baud = 115200; 
};

struct PreProcessConfig{
    int gaussian_k_size = 3;
    float gaussian_sigma = 1.3;
    float rdc_exposure_x = 0.6;
    float rdc_exposure_y = -30.0;
    float min_area = 30.0;
    float max_area = 50.0;
    float min_ratio = 2.0;
    float max_ratio = 4.0;
    float max_angle = 25.0;
    float morph_k_size = 3;
};

struct ArmorConfig{
    float max_height_diff = 4.0;    // 最大高度差
    float max_angle_diff = 30.0;    // 最大角度差
    float min_w_h_ratio  = 2.0;     // 装甲板最小宽高比
    float max_w_h_ratio  = 4.0;     // 装甲板最大宽高比
    float min_center_dist = 0.5;    // 最小中心点距离
    float max_center_dist = 3.0;    // 最大中心点距离
    float min_center_bar_ratio = 0.8;     
    float max_center_bar_ratio = 1.8;
};

struct Ballistic{
    float bulletSpeed = 28.0f;      // 弹速
    float gravity = 9.8f;           // 重力加速度
    float cameraOffsetX = 0.0f;     // 偏移X
    float cameraOffsetY = 0.0f;     // 偏移Y
    float cameraOffsetZ = 0.0f;     // 偏移Z
};

struct KalmanConfig{
    float processNoisePos = 1e-4f;   // 位置过程噪声
    float processNoiseVel = 1e-2f;    // 速度过程噪声
    float measurementNoisePos = 1e-2f; // 观测噪声
    float initialErrorCov = 1.0f;      // 初始误差协方差
    float angularVelocity = 7.33f;     // 水平转速 (rad/s)，用于参考
    float yawProcessNoisePos = 1e-2f;   // 角度过程噪声
    float yawProcessNoiseVel = 1e-1f;   // 角速度过程噪声
    float yawMeasurementNoise = 1e-1f;  // 观测噪声
};

struct YoloConfig{
    std::string model_path = "models/armor.onnx";
    float conf_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int input_width = 640;
    int input_height = 640;
    int inference_interval = 3;   // 每多少帧执行一次推理
};

// 新增 UDP 配置结构体
struct UdpConfig{
    bool enabled = false;
    string host = "127.0.0.1";
    int port = 9870;
};

class Config{
public:
    //获取单例实例
    static Config& get();

    //禁止拷贝和赋值
    Config(const Config&) = delete;
    Config& operator=(const Config&);

    //配置项
    CameraConfig camera;
    SerialConfig serial;
    PreProcessConfig preprocess;
    ArmorConfig armor;
    Ballistic ballistic;
    KalmanConfig kalman;
    UdpConfig udp;
    YoloConfig yolo;

private:
    Config();
    void loadYaml(const string& file_name);
};