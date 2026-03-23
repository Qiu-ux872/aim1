#pragma once

#include <string>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>
#include <limits>
#include <cmath>

class UdpLogger {
public:
    // 构造函数
    UdpLogger(const std::string& host, int port = 9870);
    ~UdpLogger();

    // 发送数据
    void send(double timestamp,
              double pnp_distance,
              double pnp_yaw, double pnp_pitch, double pnp_roll,
              double aim_yaw, double aim_pitch,
              double est_x, double est_y, double est_z,
              double pred_x, double pred_y, double pred_z);

    bool isOpen() const { return m_sock != -1; }

private:
    int m_sock;
    struct sockaddr_in m_addr;
    socklen_t m_addr_len;

    void sendJson(const std::string& json);
    static double safeDouble(double v); // 安全转换
};