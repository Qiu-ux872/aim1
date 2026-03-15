#pragma once

#include <string>
#include "Config.hpp"
#include "Param.hpp"

using namespace std;
using namespace cv;

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    // 打开串口（自动从 Config 读取 port 和 baud）
    bool open();

    // 关闭串口
    void close();

    // 检查串口是否已打开
    bool isOpen() const { return m_fd != -1; }

    // 发送瞄准角度（AimAngle 结构体）
    bool sendAimAngle(const AimAngle& aim);

    // 发送自定义数据（用于扩展）
    bool sendData(const uint8_t* data, size_t len);

private:
    int m_fd;                     // 串口文件描述符
    string m_port;           // 串口设备名
    int m_baudrate;               // 波特率

    // 计算校验和（异或）
    uint8_t calcChecksum(const uint8_t* data, size_t len) const;
};