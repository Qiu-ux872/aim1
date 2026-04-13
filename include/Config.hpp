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
    float rgb_gain_r = 1.0f;
    float rgb_gain_g = 1.0f;
    float rgb_gain_b = 1.0f;
};

struct SerialConfig{
    string port = "/dev/ttyUSB0";
    int baud = 115200; 
};

struct PreProcessConfig{
    int color = 1;    // 0 for red, 1 for blue
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
    float max_height_diff = 4.0;
    float max_angle_diff = 30.0;
    float min_w_h_ratio  = 2.0;
    float max_w_h_ratio  = 4.0;
    float min_center_dist = 0.5;
    float max_center_dist = 3.0;
    float min_center_bar_ratio = 0.8;     
    float max_center_bar_ratio = 1.8;
};

struct Ballistic{
    float bulletSpeed = 28.0f;
    float gravity = 9.8f;
    float cameraOffsetX = 0.0f;
    float cameraOffsetY = 0.0f;
    float cameraOffsetZ = 0.0f;
};

struct KalmanConfig{
    float processNoisePos = 1e-4f;
    float processNoiseVel = 1e-2f;
    float measurementNoisePos = 1e-2f;
    float initialErrorCov = 1.0f;
    float angularVelocity = 7.33f;
    float yawProcessNoisePos = 1e-2f;
    float yawProcessNoiseVel = 1e-1f;
    float yawMeasurementNoise = 1e-1f;
};

struct TargetSelectConfig {
    float w_center = 0.4f;
    float w_distance = 0.4f;
    float w_stability = 0.2f;
    float hysteresis = 0.1f;
    int max_tracked = 10;
    float max_distance_mm = 3000.0f;
    int lost_timeout_ms = 500;
};

struct UdpConfig{
    bool enabled = false;
    string host = "127.0.0.1";
    int port = 9870;
};

class Config{
public:
    static Config& get();
    Config(const Config&) = delete;
    Config& operator=(const Config&);

    CameraConfig camera;
    SerialConfig serial;
    PreProcessConfig preprocess;
    ArmorConfig armor;
    Ballistic ballistic;
    KalmanConfig kalman;
    UdpConfig udp;
    TargetSelectConfig target_select;

private:
    Config();
    void loadYaml(const string& file_name);
};