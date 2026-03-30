#include "YoloDetector.hpp"
#include <iostream>
#include <opencv2/dnn.hpp>   // 用于 NMS

YoloDetector::YoloDetector(const YoloConfig& cfg)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session(nullptr),
      memoryInfo(nullptr),
      inputWidth(cfg.input_width),
      inputHeight(cfg.input_height),
      confThreshold(cfg.conf_threshold),
      nmsThreshold(cfg.nms_threshold)
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(env, cfg.model_path.c_str(), sessionOptions);
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 获取输入输出节点名
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    for (size_t i = 0; i < numInputNodes; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        inputNames.push_back(name.get());
    }
    size_t numOutputNodes = session.GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        outputNames.push_back(name.get());
    }

    std::cout << "[YOLO] 模型加载成功" << std::endl;
    std::cout << "       模型路径: " << cfg.model_path << std::endl;
    std::cout << "       输入尺寸: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "       置信度阈值: " << confThreshold << ", NMS阈值: " << nmsThreshold << std::endl;
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame, float& scale, int& dw, int& dh) {
    // 计算缩放比例和填充偏移（letterbox）
    float widthScale = static_cast<float>(inputWidth) / frame.cols;
    float heightScale = static_cast<float>(inputHeight) / frame.rows;
    scale = std::min(widthScale, heightScale);
    int newWidth = static_cast<int>(frame.cols * scale);
    int newHeight = static_cast<int>(frame.rows * scale);
    dw = (inputWidth - newWidth) / 2;
    dh = (inputHeight - newHeight) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newWidth, newHeight));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, inputHeight - newHeight - dh,
                       dw, inputWidth - newWidth - dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 归一化并转换为浮点
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);
    return padded;
}

void YoloDetector::postprocess(const cv::Mat& output, float scale, int dw, int dh,
                               int imgW, int imgH, std::vector<Detection>& detections) {
    // YOLOv8 输出形状为 [1, 84, 8400]（COCO 格式，80 个类别）
    const int numClasses = 80;
    const int numDetections = output.cols;  // 8400
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int i = 0; i < numDetections; ++i) {
        const float* ptr = output.ptr<float>(0, i);
        // 获取所有类别的最大置信度
        float maxConf = 0;
        int classId = -1;
        for (int c = 0; c < numClasses; ++c) {
            float prob = ptr[4 + c];
            if (prob > maxConf) {
                maxConf = prob;
                classId = c;
            }
        }
        if (maxConf > confThreshold) {
            float cx = ptr[0];
            float cy = ptr[1];
            float w = ptr[2];
            float h = ptr[3];

            // 将归一化坐标转换回原图坐标（考虑 letterbox 缩放和填充）
            float x = (cx * inputWidth - dw) / scale;
            float y = (cy * inputHeight - dh) / scale;
            float width = (w * inputWidth) / scale;
            float height = (h * inputHeight) / scale;

            int left = static_cast<int>(x - width / 2);
            int top = static_cast<int>(y - height / 2);
            int right = static_cast<int>(x + width / 2);
            int bottom = static_cast<int>(y + height / 2);

            // 裁剪到图像范围内
            left = std::max(0, left);
            top = std::max(0, top);
            right = std::min(imgW, right);
            bottom = std::min(imgH, bottom);

            boxes.emplace_back(left, top, right - left, bottom - top);
            confidences.push_back(maxConf);
        }
    }

    // 应用 NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    detections.clear();
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = 0;   // 此处可存储实际类别 ID
        detections.push_back(det);
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (frame.empty()) return detections;

    float scale;
    int dw, dh;
    cv::Mat inputBlob = preprocess(frame, scale, dw, dh);

    // 准备输入张量
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    size_t inputTensorSize = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3];
    std::vector<float> inputTensorValues(inputBlob.begin<float>(), inputBlob.end<float>());

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputShape.data(), inputShape.size());

    // 推理
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
                               outputNames.data(), outputNames.size());

    // 获取输出张量（假设只有一个输出）
    float* outputData = outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // 输出形状通常为 [1, 84, 8400]，我们转换为 84 x 8400 的 Mat
    cv::Mat outputMat(static_cast<int>(outputShape[1]), static_cast<int>(outputShape[2]),
                      CV_32F, outputData);

    postprocess(outputMat, scale, dw, dh, frame.cols, frame.rows, detections);
    return detections;
}