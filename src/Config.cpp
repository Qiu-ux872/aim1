#include "Config.hpp"
#include <iostream>

Config& Config::get(){
    static Config instance;
    return instance;
}

Config::Config(){
    loadYaml("/config/Config.yaml");
}

void Config::loadYaml(const string& file_name){
    try{
        YAML::Node node = YAML::LoadFile(file_name);

        camera.width = node["camera"]["width"].as<int>();
        camera.height = node["camera"]["height"].as<int>();
        camera.fps = node["camera"]["fps"].as<int>();
        camera.exposure = node["camera"]["exposure"].as<float>();
        camera.gain = node["camera"]["gain"].as<float>();

        preprocess.gaussian_k_size = node["preprocess"]["gaussian_k_size"].as<int>();
        preprocess.gaussian_sigma = node["preprocess"]["gaussian_sigma"].as<float>();
        preprocess.rdc_exposure_x = node["preprocess"]["rdc_exposure_x"].as<float>();
        preprocess.rdc_exposure_y = node["preprocess"]["rdc_exposure_y"].as<float>();
        preprocess.min_area = node["preprocess"]["min_area"].as<float>();
        preprocess.max_area = node["preprocess"]["max_area"].as<float>();
        preprocess.min_ratio = node["preprocess"]["min_ratio"].as<float>();
        preprocess.max_ratio = node["preprocess"]["max_ratio"].as<float>();
        preprocess.max_angle = node["preprocess"]["max_angle"].as<float>();

        serial.port = node["serial"]["port"].as<string>();
        serial.baud = node["serial"]["baud"].as<int>();

        armor.max_height_diff = node["armor"]["max_height_diff"].as<float>();
        armor.min_dist_ratio = node["armor"]["min_dist_ratio"].as<float>();
        armor.max_dist_ratio = node["armor"]["max_dist_ratio"].as<float>();
        armor.max_angle_diff = node["armor"]["max_angle_diff"].as<float>();
        armor.w_h_ratio = node["armor"]["w_h_ratio"].as<float>();

        ballistic.bulletSpeed = node["Ballistic"]["bulletSpeed"].as<float>();
        ballistic.gravity = node["Ballistic"]["gravity"].as<float>();
        ballistic.cameraOffsetX = node["Ballistic"]["cameraOffsetX"].as<float>();
        ballistic.cameraOffsetY = node["Ballistic"]["cameraOffsetY"].as<float>();
        ballistic.cameraOffsetZ = node["Ballistic"]["cameraOffsetZ"].as<float>();

        kalman.processNoisePos = node["kalman"]["processNoisePos"].as<float>();
        kalman.processNoiseVel = node["kalman"]["processNoiseVel"].as<float>();
        kalman.measurementNoisePos = node["kalman"]["measurementNoisePos"].as<float>();
        kalman.initialErrorCov = node["kalman"]["initialErrorCov"].as<float>();
        kalman.angularVelocity = node["kalman"]["angularVelocity"].as<float>();

    } catch (const YAML::Exception& e) {
        cerr << "[Config] Yaml文件加载错误:" << e.what() << endl;
    }
}