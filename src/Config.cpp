#include "Config.hpp"
#include <iostream>

Config& Config::get(){
    static Config instance;
    return instance;
}

Config::Config(){
    loadYaml("/home/qiu/桌面/aim/config/Config.yaml");
}

void Config::loadYaml(const string& file_name){
    try{
        YAML::Node node = YAML::LoadFile(file_name);

        camera.width = node["camera"]["width"].as<int>();
        camera.height = node["camera"]["height"].as<int>();
        camera.fps = node["camera"]["fps"].as<int>();
        camera.exposure = node["camera"]["exposure"].as<float>();
        camera.gain = node["camera"]["gain"].as<float>();
        camera.rgb_gain_r = node["camera"]["rgb_gain_r"].as<float>();
        camera.rgb_gain_g = node["camera"]["rgb_gain_g"].as<float>();
        camera.rgb_gain_b = node["camera"]["rgb_gain_b"].as<float>();

        preprocess.gaussian_k_size = node["preprocess"]["gaussian_k_size"].as<int>();
        preprocess.gaussian_sigma = node["preprocess"]["gaussian_sigma"].as<float>();
        preprocess.rdc_exposure_x = node["preprocess"]["rdc_exposure_x"].as<float>();
        preprocess.rdc_exposure_y = node["preprocess"]["rdc_exposure_y"].as<float>();
        preprocess.min_area = node["preprocess"]["min_area"].as<float>();
        preprocess.max_area = node["preprocess"]["max_area"].as<float>();
        preprocess.min_ratio = node["preprocess"]["min_ratio"].as<float>();
        preprocess.max_ratio = node["preprocess"]["max_ratio"].as<float>();
        preprocess.max_angle = node["preprocess"]["max_angle"].as<float>();
        preprocess.morph_k_size = node["preprocess"]["morph_k_size"].as<int>();
        
        serial.port = node["serial"]["port"].as<string>();
        serial.baud = node["serial"]["baud"].as<int>();

        armor.max_height_diff = node["armor"]["max_height_diff"].as<float>();
        armor.max_angle_diff = node["armor"]["max_angle_diff"].as<float>();
        armor.min_center_dist = node["armor"]["min_center_dist"].as<float>();
        armor.max_center_dist = node["armor"]["max_center_dist"].as<float>();
        armor.min_w_h_ratio = node["armor"]["min_w_h_ratio"].as<float>();
        armor.max_w_h_ratio = node["armor"]["max_w_h_ratio"].as<float>();
        armor.max_center_bar_ratio = node["armor"]["max_center_bar_ratio"].as<float>();
        armor.min_center_bar_ratio = node["armor"]["min_center_bar_ratio"].as<float>();

        // 读取 ballistic 节点（注意节点名与 YAML 一致，这里使用小写）
        ballistic.bulletSpeed = node["ballistic"]["bulletSpeed"].as<float>();
        ballistic.gravity = node["ballistic"]["gravity"].as<float>();
        ballistic.cameraOffsetX = node["ballistic"]["camera_offset_x"].as<float>();
        ballistic.cameraOffsetY = node["ballistic"]["camera_offset_y"].as<float>();
        ballistic.cameraOffsetZ = node["ballistic"]["camera_offset_z"].as<float>();

        kalman.processNoisePos = node["kalman"]["processNoisePos"].as<float>();
        kalman.processNoiseVel = node["kalman"]["processNoiseVel"].as<float>();
        kalman.measurementNoisePos = node["kalman"]["measurementNoisePos"].as<float>();
        kalman.initialErrorCov = node["kalman"]["initialErrorCov"].as<float>();
        kalman.angularVelocity = node["kalman"]["angularVelocity"].as<float>();
        kalman.yawProcessNoisePos = node["kalman"]["yawProcessNoisePos"].as<float>();
        kalman.yawProcessNoiseVel = node["kalman"]["yawProcessNoiseVel"].as<float>();
        kalman.yawMeasurementNoise = node["kalman"]["yawMeasurementNoise"].as<float>();
        kalman.yawProcessNoisePos = node["kalman"]["yawProcessNoisePos"].as<float>();
        kalman.yawProcessNoiseVel = node["kalman"]["yawProcessNoiseVel"].as<float>();
        kalman.yawMeasurementNoise = node["kalman"]["yawMeasurementNoise"].as<float>();

        // 读取 UDP 配置
        if (node["udp"]) {
            udp.enabled = node["udp"]["enabled"].as<bool>();
            udp.host = node["udp"]["host"].as<string>();
            udp.port = node["udp"]["port"].as<int>();
        }

    } catch (const YAML::Exception& e) {
        cerr << "[Config] Yaml文件加载错误:" << e.what() << endl;
    }
}