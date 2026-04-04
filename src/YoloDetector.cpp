#include "YoloDetector.hpp"
#include <iostream>
#include <opencv2/dnn.hpp>

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
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(env, cfg.model_path.c_str(), sessionOptions);
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 获取输入输出节点名（仅用于调试）
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

    m_inputName = "images";
    m_outputName = "output0";

    std::cout << "[YOLO] ONNX 模型加载成功" << std::endl;
    std::cout << "       模型路径: " << cfg.model_path << std::endl;
    std::cout << "       输入尺寸: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "       置信度阈值: " << confThreshold << ", NMS阈值: " << nmsThreshold << std::endl;
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame, float& scale, int& dw, int& dh) {
    // 1. 计算 letterbox 缩放和填充
    float widthScale = static_cast<float>(inputWidth) / frame.cols;
    float heightScale = static_cast<float>(inputHeight) / frame.rows;
    scale = std::min(widthScale, heightScale);
    int newWidth = static_cast<int>(frame.cols * scale);
    int newHeight = static_cast<int>(frame.rows * scale);
    dw = (inputWidth - newWidth) / 2;
    dh = (inputHeight - newHeight) / 2;

    // 2. 缩放并填充
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newWidth, newHeight));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, inputHeight - newHeight - dh,
                       dw, inputWidth - newWidth - dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 3. 转换为 NCHW 格式并归一化 (0~1)
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0/255.0, cv::Size(), cv::Scalar(), false, false);
    // blob 的形状为 [1, 3, inputHeight, inputWidth]，数据类型为 float32
    return blob;
}

void YoloDetector::postprocess(const cv::Mat& output, float scale, int dw, int dh,
                               int imgW, int imgH, std::vector<Detection>& detections) {
    // 输出形状: [1, 6, 8400] -> 6 行，8400 列
    const int numDetections = output.cols;   // 8400
    const int numValues = output.rows;       // 6
    if (numValues != 6) {
        std::cerr << "[YOLO] 输出维度异常，期望6，实际" << numValues << std::endl;
        return;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int i = 0; i < numDetections; ++i) {
        const float* ptr = output.ptr<float>(0, i);
        float confidence = ptr[4];   // 假设置信度在第5个元素（索引4）
        if (confidence > confThreshold) {
            float cx = ptr[0];
            float cy = ptr[1];
            float w = ptr[2];
            float h = ptr[3];

            float x = (cx * inputWidth - dw) / scale;
            float y = (cy * inputHeight - dh) / scale;
            float width = (w * inputWidth) / scale;
            float height = (h * inputHeight) / scale;

            int left = static_cast<int>(x - width / 2);
            int top = static_cast<int>(y - height / 2);
            int right = static_cast<int>(x + width / 2);
            int bottom = static_cast<int>(y + height / 2);

            left = std::max(0, left);
            top = std::max(0, top);
            right = std::min(imgW, right);
            bottom = std::min(imgH, bottom);

            boxes.emplace_back(left, top, right - left, bottom - top);
            confidences.push_back(confidence);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    detections.clear();
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = 0;
        detections.push_back(det);
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (frame.empty()) return detections;

    float scale;
    int dw, dh;
    cv::Mat inputBlob = preprocess(frame, scale, dw, dh);
    // inputBlob 是 [1, 3, inputHeight, inputWidth] 的 float 张量

    // 准备输入张量
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    size_t inputTensorSize = inputBlob.total() * inputBlob.elemSize(); // 总字节数
    // 直接使用 inputBlob 的数据指针
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputBlob.ptr<float>(), inputTensorSize / sizeof(float),
        inputShape.data(), inputShape.size());

    // 推理
    const char* inputNames[] = {m_inputName.c_str()};
    const char* outputNames[] = {m_outputName.c_str()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                               outputNames, 1);

    // 获取输出张量
    float* outputData = outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    cv::Mat outputMat(static_cast<int>(outputShape[1]), static_cast<int>(outputShape[2]),
                      CV_32F, outputData);

    postprocess(outputMat, scale, dw, dh, frame.cols, frame.rows, detections);
    return detections;
}