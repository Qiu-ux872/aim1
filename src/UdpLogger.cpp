#include "UdpLogger.hpp"

UdpLogger::UdpLogger(const std::string& host, int port) {
    m_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sock < 0) {
        std::cerr << "[UDP] 创建socket失败" << std::endl;
        return;
    }
    memset(&m_addr, 0, sizeof(m_addr));
    m_addr.sin_family = AF_INET;
    m_addr.sin_port = htons(port);
    m_addr.sin_addr.s_addr = inet_addr(host.c_str());
    m_addr_len = sizeof(m_addr);
}

UdpLogger::~UdpLogger() {
    if (m_sock != -1) close(m_sock);
}

void UdpLogger::send(double timestamp,
                     double pnp_distance,
                     double pnp_yaw, double pnp_pitch, double pnp_roll,
                     double aim_yaw, double aim_pitch,
                     double est_x, double est_y, double est_z,
                     double pred_x, double pred_y, double pred_z) {
    if (m_sock == -1) return;
    std::stringstream ss;
    ss << "{"
       << "\"timestamp\":" << timestamp << ","
       << "\"pnp_distance\":" << pnp_distance << ","
       << "\"pnp_yaw\":" << pnp_yaw << ","
       << "\"pnp_pitch\":" << pnp_pitch << ","
       << "\"pnp_roll\":" << pnp_roll << ","
       << "\"aim_yaw\":" << aim_yaw << ","
       << "\"aim_pitch\":" << aim_pitch << ","
       << "\"est_x\":" << est_x << ","
       << "\"est_y\":" << est_y << ","
       << "\"est_z\":" << est_z << ","
       << "\"pred_x\":" << pred_x << ","
       << "\"pred_y\":" << pred_y << ","
       << "\"pred_z\":" << pred_z
       << "}";
    sendJson(ss.str());
}

void UdpLogger::sendJson(const std::string& json) {
    sendto(m_sock, json.c_str(), json.length(), 0,
           (struct sockaddr*)&m_addr, m_addr_len);
}