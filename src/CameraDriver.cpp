#include "Config.hpp"
#include "CameraDriver.hpp"
#include "Param.hpp"

using namespace std;
using namespace cv;

// 初始化相机
CameraDriver::CameraDriver()
    : m_hCamera(0),
    m_isOpen(false),
    m_isStreaming(false),
    m_width(640),
    m_height(489)
{
    CameraSdkInit(1);
}

CameraDriver::~CameraDriver(){
    close();
}

void CameraDriver::printError(CameraSdkStatus status, const char* msg){
    cerr << "[相机错误]" << msg << "错误码:" << status
    << "(" << CameraGetErrorString(status) << ")" << endl;
}

bool CameraDriver::open(){
    if(m_isOpen){
        cout << "相机已打开" << endl;
    }

    // 枚举设备
    tSdkCameraDevInfo devList[4];
    int count = 4;
    CameraSdkStatus status = CameraEnumerateDevice(devList, &count);
    if(status != CAMERA_STATUS_SUCCESS){
        printError(status, "枚举失败");
        return false;
    }

    if(count == 0){
        cerr << "[相机] 未找到相机" << endl;
        return false;
    }
    cout << "找到" << count << "台相机，使用第一台" << endl;
    m_devInfo = devList[0];

    // 初始化
    status = CameraInit(&m_devInfo, -1, -1, &m_hCamera);
    if(status != CAMERA_STATUS_SUCCESS){
        printError(status, "初始化失败");
        return false;
    }
    m_isOpen = true;

    // 获取相机特性
    status = CameraGetCapability(m_hCamera, &m_capability);
    if(status != CAMERA_STATUS_SUCCESS){
        printError(status, "获取特性失败");
        close();
        return false;
    }

    // 设置输出格式RGB24
    status = CameraSetIspOutFormat(m_hCamera, CAMERA_MEDIA_TYPE_BGR8);
    if(status != CAMERA_STATUS_SUCCESS){
        printError(status, "设置输出格式失败");
    }

    // 从Config读取分辨率
    const auto& cfg = Config::get();
    m_width = cfg.camera.width;
    m_height = cfg.camera.height;

    // 构建分辨率结构体（自定义分辨率）
    tSdkImageResolution resolution;
    memset(&resolution, 0, sizeof(tSdkImageResolution));
    resolution.iIndex      = 0xFF;           // 0xFF 表示自定义 ROI
    resolution.iWidthFOV   = m_width;        // 实际视场宽度
    resolution.iHeightFOV  = m_height;       // 实际视场高度
    resolution.iWidth      = m_width;        // 输出图像宽度
    resolution.iHeight     = m_height;       // 输出图像高度
    resolution.iHOffsetFOV = 0;              // 水平偏移（0 表示居中，可计算）
    resolution.iVOffsetFOV = 0;              // 垂直偏移

    status = CameraSetImageResolution(m_hCamera, &resolution);
    if (status != CAMERA_STATUS_SUCCESS) {
        printError(status, "设置分辨率失败");
        // 继续，可能当前分辨率有效
    }
    // 设置为连续采集模式（0 通常表示连续）
    CameraSetTriggerMode(m_hCamera, 0);

    // 设置曝光参数
    float exposureUs = cfg.camera.exposure;
    if(exposureUs > 0){
        // 手动曝光
        setAutoExposure(false);
        setExposureTime(exposureUs);
    } else {
        setAutoExposure(true);
    }

    // 设置模拟增益
    float gain = cfg.camera.gain;
    if(gain > 0){
        setAnalogGain(gain);
    }

    cout << "相机打开成功,分辨率：" << m_width << "x" << m_height << endl;
    return true;
}

void CameraDriver::close(){
    if(!m_isStreaming){
        stop();
    }
    if(m_isOpen){
        CameraUnInit(m_hCamera);
        m_isOpen = false;
        cout << "相机已关闭" << endl;
    }
}

bool CameraDriver::start(){
    if(!m_isOpen){
        cerr << "[相机] 请先打开相机" << endl;
        return false;
    }
    if(m_isStreaming){
        return true;
    }
    CameraSdkStatus status = CameraPlay(m_hCamera);
    if(status != CAMERA_STATUS_SUCCESS){
        printError(status, "开始采集失败");
        return false;
    }
    m_isStreaming = true;
    cout << "相机开始采集" << endl;
    return true;
}

void CameraDriver::stop(){
    if(!m_isStreaming) return;
    CameraPause(m_hCamera);
    m_isStreaming = false;
    cout << "暂停采集" << endl;
}

unsigned char* CameraDriver::captureRaw(int& width, int& height, int timeouMs){
    if(!m_isStreaming){
        cerr << "[相机] 请先开始采集" << endl;
        return nullptr;
    }

    INT w, h;
    unsigned char* pBuffer = CameraGetImageBufferEx(m_hCamera, &w, &h, timeouMs);
    if(pBuffer == nullptr){
        cerr << "获取图像超时(" << timeouMs << "ms)" << endl;
        return nullptr;
    }
    width = w;
    height = h;
    return pBuffer;
}

Mat CameraDriver::capture(int timeoutMs){
    int w, h;
    unsigned char* data = captureRaw(w, h, timeoutMs);
    if(data == nullptr){
        return Mat();
    }

    Mat image(h, w, CV_8UC3);
    memcpy(image.data, data, w * h * 3);
    return image;
}

bool CameraDriver::setAnalogGain(float gain) {
    if (!m_isOpen) return false;
    // CameraSetAnalogGainX 设置增益倍数（如 1.0, 2.0）
    CameraSdkStatus status = CameraSetAnalogGainX(m_hCamera, gain);
    if (status != CAMERA_STATUS_SUCCESS) {
        printError(status, "设置模拟增益失败");
        return false;
    }
    return true;
}

bool CameraDriver::setAutoExposure(bool enable) {
    if (!m_isOpen) return false;
    CameraSdkStatus status = CameraSetAeState(m_hCamera, enable ? TRUE : FALSE);
    if (status != CAMERA_STATUS_SUCCESS) {
        printError(status, enable ? "开启自动曝光失败" : "关闭自动曝光失败");
        return false;
    }
    return true;
}
bool CameraDriver::setExposureTime(float exposureUs) {
    if (!m_isOpen) return false;
    CameraSdkStatus status = CameraSetExposureTime(m_hCamera, exposureUs);
    if (status != CAMERA_STATUS_SUCCESS) {
        printError(status, "设置曝光时间失败");
        return false;
    }
    return true;
}