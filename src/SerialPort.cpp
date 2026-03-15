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
            cerr << "警告：不支持的波特率 " << baudrate << "，使用 115200" << endl;
            speed = B115200;
            break;
    }
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= CREAD | CLOCAL;

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_iflag &= ~(INLCR | ICRNL | IGNCR);
    options.c_oflag &= ~OPOST;

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 1;

    tcflush(m_fd, TCIOFLUSH);

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

    uint8_t buffer[10];
    buffer[0] = 0xAA;
    memcpy(buffer + 1, &aim.yaw, 4);
    memcpy(buffer + 5, &aim.pitch, 4);
    buffer[9] = calcChecksum(buffer, 9);

    ssize_t written = write(m_fd, buffer, sizeof(buffer));
    if (written != sizeof(buffer)) {
        cerr << "串口写入失败，实际写入 " << written << " 字节" << endl;
        return false;
    }
    tcdrain(m_fd);
    return true;
}

bool SerialPort::sendData(const uint8_t* data, size_t len) {
    if (m_fd == -1) return false;
    ssize_t written = write(m_fd, data, len);
    if (written != (ssize_t)len) return false;
    tcdrain(m_fd);
    return true;
}