#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "Config.hpp"

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class YoloDetector {
public:
    YoloDetector(const YoloConfig& cfg);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::Mat preprocess(const cv::Mat& frame, float& scale, int& dw, int& dh);
    void postprocess(const cv::Mat& output, float scale, int dw, int dh,
                     int imgW, int imgH, std::vector<Detection>& detections);

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memoryInfo;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::string m_inputName;
    std::string m_outputName;

    int inputWidth;
    int inputHeight;
    float confThreshold;
    float nmsThreshold;
};