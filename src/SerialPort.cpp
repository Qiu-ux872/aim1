#include "SerialPort.hpp"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>

using namespace std;
using namespace cv;

SerialPort::SerialPort() : m_fd(-1) {
    const auto& cfg = Config::get().serial;
    m_port = cfg.port;
    m_baudrate = cfg.baud;
}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open() {
    if (m_fd != -1) {
        cout << "串口已打开" << endl;
        return true;
    }

    m_fd = ::open(m_port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (m_fd == -1) {
        cerr << "无法打开串口: " << m_port << endl;
        return false;
    }

    struct termios options;
    if (tcgetattr(m_fd, &options) != 0) {
        cerr << "获取串口属性失败" << endl;
        close();
        return false;
    }

    int baudrate = m_baudrate;
    speed_t speed;
    switch (baudrate) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        case 460800: speed = B460800; break;
        default:
            cerr << "不支持的波特率 " << baudrate << "，使用 115200" << endl;
            speed = B115200;
            break;
    }
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    options.c_cflag &= ~PARENB;        // 无奇偶校验
    options.c_cflag &= ~CSTOPB;        // 1位停止位
    options.c_cflag &= ~CSIZE;         // 清除数据位设置
    options.c_cflag |= CS8;            // 8位数据位
    options.c_cflag |= CREAD | CLOCAL; // 启用接收器，忽略调制解调器控制线

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始模式，非规范输入，禁用回显
    options.c_iflag &= ~(IXON | IXOFF | IXANY);        // 禁用软件流控
    options.c_iflag &= ~(INLCR | ICRNL | IGNCR);       // 禁用输入转换
    options.c_oflag &= ~OPOST;                           // 原始输出模式

    options.c_cc[VMIN] = 0;   // 非阻塞读取，最少读取0个字符
    options.c_cc[VTIME] = 1;   // 超时设置：0.1秒

    tcflush(m_fd, TCIOFLUSH);  // 清空输入输出缓冲区

    if (tcsetattr(m_fd, TCSANOW, &options) != 0) {
        cerr << "设置串口属性失败" << endl;
        close();
        return false;
    }

    cout << "串口 " << m_port << " 打开成功，波特率 " << baudrate << endl;
    return true;
}

void SerialPort::close() {
    if (m_fd != -1) {
        ::close(m_fd);
        m_fd = -1;
        cout << "串口已关闭" << endl;
    }
}

uint8_t SerialPort::calcChecksum(const uint8_t* data, size_t len) const {
    uint8_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum ^= data[i];
    }
    return sum;
}

bool SerialPort::sendAimAngle(const AimAngle& aim) {
    if (m_fd == -1) {
        cerr << "串口未打开" << endl;
        return false;
    }

    // 文本格式：yaw,pitch,distance\n
    char buffer[128];
    int len = snprintf(buffer, sizeof(buffer), "%.4f,%.4f,%.4f\n",
                      aim.yaw, aim.pitch, aim.distance);

    if (len <= 0 || len >= (int)sizeof(buffer)) {
        cerr << "格式化数据失败" << endl;
        return false;
    }

    ssize_t written = write(m_fd, buffer, len);
    if (written != len) {
        cerr << "串口写入失败，实际写入 " << written << " 字节" << endl;
        return false;
    }
    tcdrain(m_fd);  // 等待所有数据发送完成
    return true;
}

bool SerialPort::sendData(const uint8_t* data, size_t len) {
    if (m_fd == -1) return false;
    ssize_t written = write(m_fd, data, len);
    if (written != (ssize_t)len) return false;
    tcdrain(m_fd);
    return true;
}