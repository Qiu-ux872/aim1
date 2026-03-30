#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "Config.hpp"

struct Detection {
    cv::Rect box;           // 边界框（原图像坐标）
    float confidence;       // 置信度
    int classId;            // 类别 ID（本例中为 0）
};

class YoloDetector {
public:
    YoloDetector(const YoloConfig& cfg);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // 预处理
    cv::Mat preprocess(const cv::Mat& frame, float& scale, int& dw, int& dh);
    // 后处理：解析 ONNX 输出，将坐标映射回原图，应用 NMS
    void postprocess(const cv::Mat& output, float scale, int dw, int dh,
                     int imgW, int imgH, std::vector<Detection>& detections);

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memoryInfo;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    int inputWidth;
    int inputHeight;
    float confThreshold;
    float nmsThreshold;
};