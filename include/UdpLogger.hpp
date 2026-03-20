#pragma once

#include <string>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>

class UdpLogger {
public:
    UdpLogger(const std::string& host = "127.0.0.1", int port = 9870);
    ~UdpLogger();

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
};